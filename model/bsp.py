import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import mcubes
import trimesh
import pandas as pd
import functools

from lib.bspt.bspt import get_mesh_watertight
from util.consts import *

Z_DIM = 128

class PolyMesh:
    """
    A class to represent a polygonal mesh with vertices and faces.

    Attributes:
        vertices (np.array): A numpy array of shape (N, 3) containing the coordinates of the vertices.
        faces (list): A list of lists containing the indices of the vertices that form each face.
        tmesh (trimesh.Trimesh): A trimesh object that represents the triangulated version of the polygonal mesh.

    Methods:
        export(name): Save the polygonal mesh as a PLY file with the given name.
        sort_vertex_clockwise(vs): Sort a list of coplanar 3D points in clockwise order around their centroid.
        to_trimesh(): Convert the polygonal mesh to a trimesh object by triangulating the faces.
    """
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = faces
        self.tmesh = None
    
    def export(self, name):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(self.vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(self.faces))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(self.vertices)):
            fout.write(str(self.vertices[ii][0])+" "+str(self.vertices[ii][1])+" "+str(self.vertices[ii][2])+"\n")
        for ii in range(len(self.faces)):
            fout.write(str(len(self.faces[ii])))
            for jj in range(len(self.faces[ii])):
                fout.write(" "+str(self.faces[ii][jj]))
            fout.write("\n")
        fout.close()
    
    def sort_vertex_clockwise(self, vs):
        # vs: [N, 3], a list of coplanar 3D points.
        assert vs.shape[0] >= 3
        # calculate center
        c = vs.mean(0)
        # calculate plane normal
        n = np.cross(vs[0]-c, vs[1]-c)
        # argsort counterclockwise
        indices = sorted(range(vs.shape[0]), key=functools.cmp_to_key(lambda i,j: np.dot(n, np.cross(vs[i]-c, vs[j]-c))))
        
        return indices

    def to_trimesh(self):
        # not lazy
        #if self.tmesh is None:
        if True:
            # triangulate polygons
            triangles = []
            # each face contains 3+ points
            for face in self.faces:
                if len(face) == 3:
                    triangles.append(face)
                else:
                    # split a 3d polygon into 3d triangles.
                    vs = []
                    for v in face: 
                        vs.append(self.vertices[v])
                    vs = np.stack(vs, 0)
                    inds = self.sort_vertex_clockwise(vs)
                    for iind in range(1, len(face)-1):
                        triangles.append([face[inds[0]], face[inds[iind]], face[inds[iind+1]]])
            self.tmesh = trimesh.Trimesh(self.vertices, np.array(triangles), process=False)
        return self.tmesh

def make_coord(shape, ranges=None, flatten=True):
    """
    Create a tensor of coordinates at the center of each grid cell.

    Args:
        shape (list of int): The shape of the grid, e.g. [H, W].
        ranges (list of tuple, optional): The ranges of the coordinates in each dimension, e.g. [(-1, 1), (-2, 2)]. If None, use [-1, 1] for all dimensions. Defaults to None.
        flatten (bool, optional): Whether to flatten the output tensor to have shape [num_cells, num_dims] or keep it as [H, W, num_dims]. Defaults to True.

    Returns:
        torch.Tensor: The coordinate tensor.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret

class generator(nn.Module):
    """
    A neural network that generates convex and concave regions from planes and points.

    Attributes:
        p_dim (int): The number of planes.
        c_dim (int): The number of convex regions.
        convex_layer_weights (torch.nn.Parameter): A matrix of shape [p_dim, c_dim] that determines how the planes are combined to form convex regions.
        concave_layer_weights (torch.nn.Parameter): A matrix of shape [c_dim, 1] that determines how the convex regions are combined to form concave regions.
    """
    def __init__(self, p_dim, c_dim):
        """
        Initialize the generator with the number of planes and convex regions.

        Args:
            p_dim (int): The number of planes.
            c_dim (int): The number of convex regions.
        """
        super(generator, self).__init__()

        self.p_dim = p_dim
        self.c_dim = c_dim

        # same for all observations. (thsu the key is plane learning, not combination learning)
        self.convex_layer_weights = nn.Parameter(torch.zeros((self.p_dim, self.c_dim)))
        self.concave_layer_weights = nn.Parameter(torch.zeros((self.c_dim, 1)))

        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

    def forward(self, points, planes, phase=0):
        """
        Perform the forward pass of the network.

        Args:
            points (torch.Tensor): A tensor of shape [B, N, 3] or [B, N, 4] where B is the batch size, N is the number
            of points, and 3 or 4 are the coordinates of each point (the last coordinate is 1 if the points are in homogeneous coordinates).
            planes (torch.Tensor): A tensor of shape [B, 4, P] where P is the number of planes and 4 are the coefficients of each plane equation.
            phase (int, optional): An integer that indicates the mode of operation of the network. Defaults to 0.
                - 0: continuous pretraining (S+)
                - 1: discrete (S*)
                - 2: discrete + overlap loss (S*)
                - 3: soft discrete
                - 4: soft discrete + overlap loss

        Returns:
            tuple of torch.Tensor: A tuple of two tensors: h2 and h3.
                - h2 (torch.Tensor): A tensor of shape [B, N, C] where C is the number of convex regions and each element indicates whether the corresponding point is inside the corresponding convex region.
                - h3 (torch.Tensor): A tensor of shape [B, N, 1] and each element indicates whether the corresponding point is inside the concave region formed by the convex regions.
        """
        # 0: continuous pretraining (S+)
        # 1: discrete (S*)
        # 2: discrete + overlap loss (S*)
        # 3: soft discrete
        # 4: soft discrete + overlap loss

        # # to homo
        if points.shape[-1] == 3:
            points = torch.cat([points, torch.ones(points.shape[0], points.shape[1], 1).to(points.device)], dim=-1)  # to homo

        # return h2, h3
        #level 1
        h1 = torch.matmul(points, planes) # [B, N, 4] x [B, 4, P] -> [B, N, P], if the point is in the correct side of the plane
        h1 = torch.clamp(h1, min=0)
        if phase==0:

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights) # [B, N, C], if the point is in the convex
            h2 = torch.clamp(1-h2, min=0, max=1)

            #level 3
            h3 = torch.matmul(h2, self.concave_layer_weights) # [B, N, 1], if the point is in the concave (final shape)
            h3 = torch.clamp(h3, min=0, max=1)

            return h2, h3
        elif phase==1 or phase==2:

            #level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights > 0.01).float())

            #level 3
            h3 = torch.min(h2, dim=2, keepdim=True)[0]

            return h2,h3
        elif phase==3 or phase==4:

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)

            #level 3
            h3 = torch.min(h2, dim=2, keepdim=True)[0]

            return h2, h3

class decoder(nn.Module):
    """
    A decoder class that inherits from the nn.Module class.

    The decoder takes a tensor of latent vectors and labels (zs_labels) as input and outputs a tensor of plane
    coefficients (l4) for each shape. The decoder consists of four linear layers with batch normalization and leaky
    ReLU activation functions. The linear layers transform the input tensor to a higher-dimensional feature space and
    then reduce it to the desired output dimension. The output tensor has a shape of [batch_size, 4, p_dim],
    where 4 is the number of plane coefficients and p_dim is the number of planes. The decoder uses Xavier uniform
    initialization for the weights and zero initialization for the biases of the linear layers.

    Attributes:
        ef_dim (int): The dimension of the feature space.
        num_classes (int): The number of shape classes.
        p_dim (int): The number of planes.

        linear_1 (nn.Linear): The first linear layer.
        linear_2 (nn.Linear): The second linear layer.
        linear_3 (nn.Linear): The third linear layer.
        linear_4 (nn.Linear): The fourth linear layer.
        bn_1 (nn.BatchNorm1d): The first batch normalization layer.
        bn_2 (nn.BatchNorm1d): The second batch normalization layer.
        bn_3 (nn.BatchNorm1d): The third batch normalization layer.

    Methods:
        forward(zs_labels): Performs the forward pass of the decoder and returns the output tensor of plane coefficients.
    """
    def __init__(self, ef_dim, num_classes, p_dim):
        super(decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.num_classes = num_classes
        self.linear_1 = nn.Linear(Z_DIM + self.num_classes, self.ef_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
        self.bn_1 = nn.BatchNorm1d(self.ef_dim*16)
        self.bn_2 = nn.BatchNorm1d(self.ef_dim*32)
        self.bn_3 = nn.BatchNorm1d(self.ef_dim*64)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias,0)


    def forward(self, zs_labels):

        l1 = self.linear_1(zs_labels)
        l1 = self.bn_1(l1)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = self.bn_2(l2)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = self.bn_3(l3)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4

def reparametrize(mu, log_sigma):
    """
    A function that performs the reparametrization trick.

    The reparametrization trick is a technique used in variational autoencoders to sample latent vectors from a normal distribution with given mean and standard deviation parameters. The function takes two arguments: mu and log_sigma, which are tensors of mean and log variance values, respectively. The function computes the standard deviation by taking the exponential of half of the log variance, and then generates a random tensor of the same shape as mu from a standard normal distribution. The function then returns a tensor of sampled latent vectors by adding the product of the random tensor and the standard deviation to the mean.

    Args:
        mu (torch.Tensor): A tensor of mean values with shape [batch_size, Z_DIM], where Z_DIM is the dimension of the latent vector.
        log_sigma (torch.Tensor): A tensor of log variance values with shape [batch_size, Z_DIM], where Z_DIM is the dimension of the latent vector.

    Returns:
        torch.Tensor: A tensor of sampled latent vectors with shape [batch_size, Z_DIM], where Z_DIM is the dimension of the latent vector.
    """
    stddev = 0.5 * torch.exp(log_sigma)
    basis = torch.randn(mu.shape).to(mu.device)
    return mu + basis * stddev

import os
from sklearn.neighbors import KDTree


def _hyperplane_project(xs, p):
    """
    A function that performs a hyperplane projection.

    The function takes two arguments: xs and p, which are numpy arrays of coordinates.
    The xs argument is a list of arrays that represent the points that define the subspace, and the p argument is an
    array that represents the point to be projected. The function computes the projection of p to the subspace defined
    by xs using matrix operations, such as subtraction, stacking, multiplication and inversion. The function returns
    an array that represents the projected point.

    Args:
        xs (list of np.array): A list of arrays of shape [N], where N is the dimension of the space, that represent the points that define the subspace.
        The length of the list should be between 2 and N, and the points should be linearly independent.
        p (np.array): An array of shape [N], where N is the dimension of the space, that represents the point to be projected.

    Returns:
        np.array: An array of shape [N], where N is the dimension of the space, that represents the projected point.
    """
    # p: [N]
    # xs: list of [N], 2 <= len <= N
    # return: projection of p to the subspace defined by xs
    x0 = xs[0]
    vs = [x - x0 for x in xs[1:]]
    A = np.stack(vs, axis=1) # [N, k]
    proj = A @ np.linalg.inv(A.T @ A) @ A.T @ (p - x0)

    return proj + x0

class CompNet(nn.Module):
    """
    A CompNet class that inherits from the nn.Module class.

    The CompNet is a neural network that generates 3D shapes from latent vectors and labels using a decoder and a generator.
    The code also loads ground truth latent vectors and meshes from a database file and builds KDTree objects for nearest
    neighbor search.

    Attributes:
        args (object): An object that contains various parameters for the network, such as num_feats, num_classes, num_planes
            and num_convexes.
        decoder (decoder): A submodule that is an instance of the decoder class, which takes latent vectors and labels as input
            and outputs plane coefficients for each shape.
        generator (generator): A submodule that is an instance of the generator class, which takes plane coefficients
            and labels as input and outputs convex masks for each shape.
        db (np.array): An array that contains ground truth latent vectors, labels and names of 3D shapes from the ScanNet dataset.
            The array has fields ‘zs’, ‘lbls’ and ‘names’.
        shapenet_path (str): A path to another dataset, ShapeNetv2, which contains watertight and simplified 3D meshes of
            various categories.
        kdtrees (dict): A dictionary that stores KDTree objects for each shape category. A KDTree object is a data structure
            that allows fast nearest neighbor search in high-dimensional spaces.
        gtzs (dict): A dictionary that stores ground truth latent vectors for each shape category.
        gtnames (dict): A dictionary that stores ground truth names for each shape category.

    Methods:
        forward(zs_labels): Performs the forward pass of the network and returns the output tensors of plane coefficients and convex masks.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.decoder = decoder(args.num_feats, args.num_classes, args.num_planes)
        self.generator = generator(args.num_planes, args.num_convexes)

        ### load gt zs and gt meshes.
        self.db = np.load(os.path.join('datasets/bsp/', 'database_scannet.npz'))
        self.shapenet_path = "datasets/ShapeNetv2_data/watertight_scaled_simplified/"

        # build KDTree
        self.kdtrees = {}
        self.gtzs = {}
        self.gtnames = {}
        for lbl in range(8):
            mask = self.db['lbls'] == lbl
            X = self.db['zs'][mask, :]
            self.gtzs[lbl] = X
            self.kdtrees[lbl] = KDTree(X[:, :Z_DIM])
            self.gtnames[lbl] = self.db['names'][mask]
            #print(f'[INFO] build KDTree for {lbl} with input {X.shape}')

    # loss
    def calculate_loss(self, convexes, valid_mask, pred_occs, occs, W_cvx, W_aux):
        """
        A method that calculates the loss function for a neural network that generates 3D shapes from latent vectors and
        labels using a convex layer.

        The method takes six arguments: convexes, valid_mask, pred_occs, occs, W_cvx and W_aux. These are tensors that
        represent the network output (convex layer), the validity mask, the network output (final output),
        the ground truth inside-outside value for each point, the connections T and the auxiliary weights W, respectively.
        The method checks the value of the attribute self.phase, which indicates the current phase of the network training.
        Depending on the value of self.phase, the method calculates the loss function differently using different terms and
        formulas. The possible values of self.phase are 0, 1, 2, 3 or 4. The method returns a dictionary that contains two keys:
        ‘loss’ and ‘loss_sp’. The values of these keys are tensors that represent the total loss and the reconstruction loss, respectively.

        Args:
            convexes (torch.Tensor): A tensor of shape [B, K, N], where B is the batch size, K is the number of convexes and N is the number of points.
                It represents the network output (convex layer), which contains plane coefficients for each convex for each shape.
            valid_mask (torch.Tensor): A tensor of shape [B], where B is the batch size. It represents a mask that indicates which shapes
                in the batch are valid and which are padded.
            pred_occs (torch.Tensor): A tensor of shape [B, 1, N], where B is the batch size and N is the number of points.
                It represents the network output (final output), which contains inside-outside values for each point for each shape.
            occs (torch.Tensor): A tensor of shape [B, 1, N], where B is the batch size and N is the number of points.
                It represents the ground truth inside-outside value for each point for each shape.
            W_cvx (torch.Tensor): A tensor of shape [B, K], where B is the batch size and K is the number of convexes.
                It represents the connections T between convexes for each shape.
            W_aux (torch.Tensor): A tensor of shape [B], where B is the batch size. It represents the auxiliary weights W for each shape.

        Returns:
            dict: A dictionary that contains two keys: ‘loss’ and ‘loss_sp’. The values of these keys are tensors that represent
            the total loss and the reconstruction loss, respectively.
        """
        # convexes - network output (convex layer), the last dim is the number of convexes
        # valid_mask: [B]
        # pred_occs - network output (final output), [B, 1, N]
        # occs - ground truth inside-outside value for each point, [B, 1, N]
        # W_cvx - connections T
        # W_aux - auxiliary weights W

        B = valid_mask.shape[0]
        pred_occs = pred_occs.view(B, -1)
        occs = occs.view(B, -1)

        if self.phase==0:
            # phase 0 continuous for better convergence
            # L_recon + L_W + L_T

            loss_sp = torch.mean((occs - pred_occs)**2, dim=1) # [B,]
            loss_sp = (loss_sp * valid_mask).sum() / (valid_mask.sum() + 1e-5)

            loss = loss_sp + torch.sum(torch.abs(W_aux-1)) + (torch.sum(torch.clamp(W_cvx-1, min=0) - torch.clamp(W_cvx, max=0)))

        elif self.phase==1:
            # phase 1 hard discrete for bsp
            # L_recon
            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)), dim=1)
            loss_sp = (loss_sp * valid_mask).sum() / (valid_mask.sum() + 1e-5)
            loss = loss_sp

        elif self.phase==2:
            # phase 2 hard discrete for bsp with L_overlap
            # L_recon + L_overlap
            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)), dim=1)
            loss_sp = (loss_sp * valid_mask).sum() / (valid_mask.sum() + 1e-5)

            G2_inside = (convexes < 0.01).float()
            bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True)>1).float()
            loss = loss_sp - torch.mean(convexes*occs*bmask)

        elif self.phase==3:
            # phase 3 soft discrete for bsp
            # L_recon + L_T
            # soft cut with loss L_T: gradually move the values in T (W_cvx) to either 0 or 1
            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)), dim=1)
            loss_sp = (loss_sp * valid_mask).sum() / (valid_mask.sum() + 1e-5)

            loss = loss_sp + torch.sum((W_cvx<0.01).float()*torch.abs(W_cvx)) + torch.sum((W_cvx>=0.01).float()*torch.abs(W_cvx-1))

        elif self.phase==4:
            #phase 4 soft discrete for bsp with L_overlap
            #L_recon + L_T + L_overlap
            #soft cut with loss L_T: gradually move the values in T (W_cvx) to either 0 or 1

            loss_sp = torch.mean((1-occs)*(1-torch.clamp(pred_occs, max=1)) + occs*(torch.clamp(pred_occs, min=0)), dim=1)
            loss_sp = (loss_sp * valid_mask).sum() / (valid_mask.sum() + 1e-5)

            G2_inside = (convexes<0.01).float()
            bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True)>1).float()
            loss = loss_sp + torch.sum((W_cvx<0.01).float()*torch.abs(W_cvx)) + torch.sum((W_cvx>=0.01).float()*torch.abs(W_cvx-1)) - torch.mean(convexes*occs*bmask)

        return {
            'loss': loss,
            'loss_sp': loss_sp,
        }

    def forward(self, data, phase=0, return_mesh='generate', projection_k=0):
        """
        Forward pass of the model.

        Args:
            data (dict): A dictionary containing the input data for the model.
                - 'bboxes': A tensor of shape [B, 6] containing the bounding boxes of the objects.
                - 'feats': A tensor of shape [B, 256] containing the latent features of the objects.
                - 'labels': A tensor of shape [B] containing the class labels of the objects.
            phase (int): The phase of the model, 0 for training and 1 for testing. Default: 0
            return_mesh (str): The mode of returning the mesh output. Can be 'generate', 'reconstruct' or 'both'. Default: 'generate'
            projection_k (int): The number of nearest neighbors to use for projecting the latent features to the hyperplane. Default: 0

        Returns:
            res (dict): A dictionary containing the output of the model.
                - 'planes': A tensor of shape [B, 4, P] containing the plane parameters of the generated or reconstructed meshes.
                - Other keys depending on the return_mesh mode.
        """

        # set phase
        self.phase = phase

        bboxes = data['bboxes'] # [B, 6]
        feats = data['feats'] # zs, [B, 256]
        labels = data['labels'] # [B, ], labels

        B = feats.shape[0]

        labels_feats = F.one_hot(labels, self.args.num_classes).float().to(feats.device) # [B, 8]

        # optimize z by project into hyperplane
        if projection_k > 0:

            feats_proj = []

            for i in range(B):
                feat = feats[i].detach().cpu().numpy() # [C]

                lbl = int(labels[i].item())
                if lbl == -1:
                    feats_proj.append(feat)
                else:
                    # topk nearest neighbor retrieval query
                    topk_ids = self.kdtrees[lbl].query(feat[None, :Z_DIM], k=projection_k, return_distance=False)[0]
                    topk_feats = self.gtzs[lbl][topk_ids] # [k, 256]

                    # literally nearest
                    if projection_k == 1:
                        feat_pro = topk_feats[0]
                    # project to hyperplane
                    else:
                        # project z to the hyper-plane defined by topk zs.
                        feat_pro = _hyperplane_project(topk_feats, feat)

                    feats_proj.append(feat_pro)

            feats_proj = np.stack(feats_proj, axis=0)
            feats_proj = torch.from_numpy(feats_proj).float().to(feats.device)
        else:
            feats_proj = feats

        ### if test, no logvar.
        if not self.args.sample:
            zs = feats_proj[:, :Z_DIM] # [B, 128]
        # randomize
        else:
            zs = reparametrize(feats_proj[:, :Z_DIM], feats_proj[:, Z_DIM:])

        zs_labels = torch.cat([zs, labels_feats], dim=1) # [B, 256+8]

        planes = self.decoder(zs_labels) # [B, 4, P]

        ### prepare results
        res = {}

        if return_mesh == 'retrieve':
            meshes = []
            mesh_ids = []
            # match zs with nearest GT zs, then use the corresponding GT mesh
            for i in range(B):

                bbox = bboxes[i]
                z = zs[i].detach().cpu().numpy() # [C]
                lbl = int(labels[i].detach().cpu().numpy())
                shapenet_id = '0' + CAD2ShapeNetID[lbl]

                # nearest neighbor query
                nnid = self.kdtrees[lbl].query(z[None, :], k=1, return_distance=False)[0, 0]

                # load gt mesh
                mesh_file = os.path.join(self.shapenet_path, shapenet_id, self.gtnames[lbl][nnid].replace('.npz', '.off'))
                mesh = trimesh.load(mesh_file, process=False)

                points = mesh.vertices

                # swap axes
                transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                points = points.dot(transform_m.T)

                # recenter + rescale
                min_xyz = points.min(0)
                max_xyz = points.max(0)
                points = points - (max_xyz + min_xyz) / 2
                points = points / (max_xyz - min_xyz) * bbox[3:6]

                # rotate
                orientation = bbox[6]
                axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
                points = points.dot(axis_rectified)

                # translate
                points = points + bbox[0:3]

                mesh.vertices = points
                meshes.append(mesh)
                mesh_ids.append(f'{shapenet_id}/{self.gtnames[lbl][nnid].replace(".npz", "")}')

                print(f'[BSP] retrieved mesh {shapenet_id}/{self.gtnames[lbl][nnid]}')

            res['meshes'] = np.array(meshes)
            res['mesh_ids'] = np.array(mesh_ids)

        if return_mesh == 'generate':
            # loop each proposal and extract mesh (slow...)
            meshes = []
            for i in range(B):

                print(f"[BSP] generating mesh {i} / {B} | {CAD_labels[labels[i].item()]} | z = {(feats[i]).abs().mean()}")

                if self.args.mesh_gen == 'mcubes':

                    if self.phase == 0: # 1 means occ
                        threshold = 0.5
                        voxels = self.implicit_to_voxels(planes[i], self.args.mise_resolution_0, self.args.mise_upsampling_steps, threshold) # [H, W, D]
                    else:
                        raise NotImplementedError

                    mesh = self.voxels_to_mesh(voxels, bbox=bboxes[i], threshold=threshold)

                elif self.args.mesh_gen == 'bspt':
                    assert self.phase == 1
                    bsp = self.implicit_to_bsp(planes[i], self.args.mise_resolution_0)
                    mesh = self.bsp_to_mesh(bsp, planes[i], bbox=bboxes[i])

                else:
                    raise NotImplementedError(self.args.mesh_gen)

                meshes.append(mesh)

            res['meshes'] = np.array(meshes)

        return res

    # query convexes, only support naive dense query
    def implicit_to_bsp(self, planes, resolution_0=64):
        """
        Converts the plane parameters to a binary space partitioning (bsp) representation.

        Args:
            planes (torch.Tensor): A tensor of shape [B, 4, P] containing the plane parameters of the meshes.
            resolution_0 (int): The resolution of the 3D grid. Default: 64

        Returns:
            bsp (torch.Tensor): A tensor of shape [H, W, D, C] containing the bsp representation of the meshes.
        """

        points = make_coord([resolution_0]*3).to(planes.device) # [HWD, 3]

        # phase must be 1, so 0 == occ
        bsp, _ = self.generator(points.unsqueeze(0), planes.unsqueeze(0), phase=1)
        bsp = bsp.view([resolution_0]*3 + [bsp.shape[-1]]) # [1, N, C] --> [H, W, D, C]

        return bsp

    # use bsp-tree to extract mesh
    def bsp_to_mesh(self, bsp, planes, bbox=None):
        """
        Converts the bsp representation to a polygon mesh.

        Args:
            bsp (torch.Tensor or np.ndarray): A tensor or array of shape [H, W, D, C] containing the bsp representation of the meshes.
            planes (torch.Tensor or np.ndarray): A tensor or array of shape [4, P] containing the plane parameters of the meshes.
            bbox (torch.Tensor or np.ndarray, optional): A tensor or array of shape [7] containing the bounding box of the mesh. Default: None

        Returns:
            mesh (PolyMesh): A PolyMesh object containing the vertices and polygons of the mesh.
        """
        # bsp: [H, W, D, C]
        # planes: [4, P]

        if torch.is_tensor(bbox):
            bbox = bbox.detach().cpu().numpy()

        # [cvx1, cvx2, ...], cvx1 = [[a1,b1,c1,d1], ...]
        bsp_convexes = []

        if torch.is_tensor(bsp):
            bsp = bsp.detach().cpu().numpy()

        # if the point is inside the convex.
        bsp = (bsp < 0.01).astype(np.int32)
        bsp_sum = bsp.sum(axis=3) # [H, W, D]

        W_cvx = self.generator.convex_layer_weights.detach().cpu().numpy() # [P, C]

        for i in range(self.args.num_convexes):
            cvx = bsp[:,:,:,i]
            if np.max(cvx) > 0: # if at least one voxel is inside this convex
                # if this convex is redundant, i.e. the convex is inside the shape
                if np.min(bsp_sum - cvx * 2) >= 0: # TODO: why 2 * cvx ?
                    bsp_sum = bsp_sum - cvx
                # add convex (by finding planes composing this convex)
                else:
                    cvx_planes = []
                    for j in range(self.args.num_planes):
                        # the j-th plane is part of the i-th convex.
                        if W_cvx[j, i] > 0.01:
                            cvx_planes.append((-planes[:, j]).tolist())
                    if len(cvx_planes) > 0:
                        bsp_convexes.append(np.array(cvx_planes, np.float32))


        vertices, polygons = get_mesh_watertight(bsp_convexes) # not a tri-angular mesh !!!, [(3), ...], [(>=3), ...]
        vertices = np.array(vertices)

        # fit back into world bbox if possible
        if bbox is not None:
            # 7-deg bbox
            center, size, angle = bbox[0:3], bbox[3:6], bbox[6]
            if vertices.shape[0] != 0:
                vertices = vertices / 2 * size
                vertices = vertices.dot(np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
                vertices = vertices + center

        # create mesh
        mesh = PolyMesh(vertices, polygons)

        return mesh