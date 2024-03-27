import argparse
import yaml
import os


def get_parser():
    """
    Parses command line arguments and returns a configuration object.

    Returns:
    argparse.Namespace: An object that holds the values of all the command line arguments.
                        This object is used as a configuration object in the program.
    """
    parser = argparse.ArgumentParser(description='RfS-Net')
    parser.add_argument('--config', type=str, default='config/rfs_pretrained_scannet.yaml', help='path to config file')
    parser.add_argument('--test_epoch', type=str, default='config/rfs_pretrained_scannet.yaml', help='path to config file')
    ### pretrain
    parser.add_argument('--pretrain', type=str, default='./pointgroup_phase2_scannet-000000256.pth', help='path to pretrained checkpoint')
    parser.add_argument('--file', type=str, default='', help='path to pcd cloud for inference')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg

# Get the configuration object by parsing the command line arguments
cfg = get_parser()

# Add an 'exp_path' attribute to the configuration object. This attribute holds the path to the experiment directory.
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
