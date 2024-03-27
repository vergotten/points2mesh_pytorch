import numpy as np
import open3d as o3d
import copy        
from scipy.spatial.transform import Rotation as sciRot


# Array of 20 colors for visualization
COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]]) / 255


def draw_numpy_pc(xs):
    """
    Visualizes point clouds using Open3D.

    Parameters:
    xs (list of ndarray): A list of point clouds to visualize. Each point cloud is an ndarray of shape [N, 3].
    """
    # x: [N, 3], ndarray
    ps = []
    for i, x in enumerate(xs):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(x)
        p.paint_uniform_color(COLOR20[i % 20])
        ps.append(p)

    # coordinate frames
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

    o3d.visualization.draw_geometries([mesh_frame] + ps)


def draw_registration_result(source, target, pc, transformation):
    """
    Visualizes the result of a registration.

    Parameters:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    pc (o3d.geometry.PointCloud): The point cloud to be transformed.
    transformation (ndarray): The transformation matrix.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    pc_temp = copy.deepcopy(pc)

    source.paint_uniform_color([1, 0, 0])
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    pc_temp.paint_uniform_color([1, 1, 0])

    #pd.paint_uniform_color([0.5, 0.5, 0.5])

    source_temp.transform(transformation)
    pc_temp.transform(transformation)

    o3d.visualization.draw_geometries([source, source_temp, target_temp, pc_temp])

def icp(a, b, c):
    """
    Performs Iterative Closest Point (ICP) registration.

    Parameters:
    a (ndarray): The source point cloud of shape [N, 3].
    b (ndarray): The target point cloud of shape [M, 3].
    c (ndarray): The point cloud to be transformed.

    Returns:
    tuple: A tuple containing the transformed point cloud and the fitness score of the registration.
    """
    # a: source pc, [N, 3]
    # b: target pc, [M, 3]
    # return: c: the pc to be transformed
    

    # to o3d pcd
    pa = o3d.geometry.PointCloud()
    pa.points = o3d.utility.Vector3dVector(a)
    #pa.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pb = o3d.geometry.PointCloud()
    pb.points = o3d.utility.Vector3dVector(b)
    #pb.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(c)
	
    # max allowed distance between paired points
    threshold = 0.04 # 0.02

    trans_init = np.eye(4)
	
    # vis
    evaluation = o3d.registration.evaluate_registration(pa, pb, threshold, trans_init)
    #print('[ICP before]', evaluation)
    #draw_registration_result(pa, pb, trans_init)

    # call icp
    #print("Apply point-to-point ICP")
    reg = o3d.registration.registration_icp(
        pa, pb, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        #o3d.registration.TransformationEstimationPointToPlane(), # need normals
        o3d.registration.ICPConvergenceCriteria(max_iteration=300))

    #print('[ICP after]', reg, reg.fitness)

    #print("Transformation is:")
    #print(reg.transformation)
    #print("")

    #draw_registration_result(pa, pb, pc, reg.transformation)

    M = reg.transformation.copy() # np.array, [4, 4]

    # decompose rotation
    
    rx = np.arctan2(M[2, 1], M[2, 2])
    ry = np.arctan2(-M[2, 0], np.sqrt(M[2, 1]**2 + M[2, 2]**2))
    rz = np.arctan2(M[1, 0], M[0, 0])

    # apply transform
    # reject if rotx or roty is too large
    if not (abs(rx) > np.pi / 18 or abs(ry) > np.pi / 18):
        pc.transform(M)

    return np.asarray(pc.points), reg.fitness


if __name__ == '__main__':
    print('ok')
