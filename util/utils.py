import torch, glob, os, numpy as np
import sys
sys.path.append('../')

from util.log import logger

class AverageMeter(object):
    """
    Class to compute and store the average and current value.
    """
    def __init__(self):
        """
        Initializes all the variables to 0.
        """
        self.reset()

    def reset(self):
        """
        Resets all the variables to 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the variables with the new value.

        Parameters:
        val (float): The new value to update.
        n (int, optional): The weight of the new value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """
    Sets the learning rate to the base LR decayed by 10 every step epochs.

    Parameters:
    optimizer (torch.optim.Optimizer): The optimizer for which the learning rate needs to be set.
    base_lr (float): The base learning rate.
    epoch (int): The current epoch.
    step_epoch (int): The step size in epochs.
    multiplier (float, optional): The multiplier for the learning rate decay. Defaults to 0.1.
    clip (float, optional): The minimum learning rate. Defaults to 1e-6.
    """
    lr = max(base_lr * (multiplier ** (epoch / step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    """
    Computes the intersection and union of the output and target.

    Parameters:
    output (np.ndarray): The output array.
    target (np.ndarray): The target array.
    K (int): The number of classes.
    ignore_index (int, optional): The index to ignore in the computation. Defaults to 255.

    Returns:
    tuple: A tuple containing the intersection, union, and target areas.
    """
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f=''):
    """
    Restores the model from a checkpoint.

    Parameters:
    model (torch.nn.Module): The model to restore.
    exp_path (str): The path to the experiment directory.
    exp_name (str): The name of the experiment.
    use_cuda (bool, optional): Whether to use CUDA. Defaults to True.
    epoch (int, optional): The epoch to restore. If 0, restores the latest checkpoint. Defaults to 0.
    dist (bool, optional): Whether the model is a distributed model. Defaults to False.
    f (str, optional): The path to the checkpoint file. If empty, finds the checkpoint file based on the epoch. Defaults to ''.

    Returns:
    int: The next epoch.
    """
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f), f
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)
        checkpoint = torch.load(f)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            ### tmp
            model.load_state_dict(checkpoint, strict=False)

    if use_cuda:
        model.cuda()
    return epoch + 1


def is_power2(num):
    """
    Checks if a number is a power of 2.

    Parameters:
    num (int): The number to check.

    Returns:
    bool: True if the number is a power of 2, False otherwise.
    """
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    """
    Checks if a number is a multiple of another number.

    Parameters:
    num (int): The number to check.
    multiple (int): The number to check divisibility against.

    Returns:
    bool: True if num is a multiple of multiple, False otherwise.
    """
    return num != 0 and num % multiple == 0


def checkpoint_save(model, exp_path, exp_name, epoch, save_freq=16, use_cuda=True):
    """
    Saves the model to a checkpoint.

    Parameters:
    model (torch.nn.Module): The model to save.
    exp_path (str): The path to the experiment directory.
    exp_name (str): The name of the experiment.
    epoch (int): The current epoch.
    save_freq (int, optional): The frequency of saving checkpoints. Defaults to 16.
    use_cuda (bool, optional): Whether to use CUDA. Defaults to True.
    """
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    logger.info('Saving ' + f)
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_model_param(model, pretrained_dict, prefix=""):
    """
    Loads parameters from a pretrained model.

    Parameters:
    model (torch.nn.Module): The model to load parameters into.
    pretrained_dict (dict): The dictionary of parameters from the pretrained model.
    prefix (str, optional): The prefix for the parameter names. Defaults to "".

    Returns:
    tuple: A tuple containing the number of parameters loaded and the total number of parameters.
    """
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    """
    Writes points and colors to an .obj file.

    Parameters:
    points (np.ndarray): The points to write.
    colors (np.ndarray): The colors to write.
    out_filename (str): The name of the output file.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    """
    Gets the offsets for each batch.

    Parameters:
    batch_idxs (torch.Tensor): The batch indices.
    bs (int): The batch size.

    Returns:
    torch.Tensor: The batch offsets.
    """
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    """
    Prints an error message and exits the program.

    Parameters:
    message (str): The error message to print.
    user_fault (bool, optional): Whether the error was caused by the user. Defaults to False.
    """
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
      sys.exit(2)
    sys.exit(-1)

