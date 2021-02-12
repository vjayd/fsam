import yaml
import torch
from torch import optim
from models.densenet_161 import DeepPixBis
from models.resnet_18 import Resnet_18, Resnet_152, Densenet_201, Facenet, Ycbcr


def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg


def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer


def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device("cpu")
    elif cfg['device'] == 'gpu':
        device = torch.device("cuda:0")
    else:
        raise NotImplementedError
    return device


def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    
    network = None

    if cfg['model']['base'] == 'densenet_161':
        network = DeepPixBis(pretrained=cfg['model']['pretrained'])
    elif cfg['model']['base'] == 'resnet_18':
        network = Resnet_18(pretrained=cfg['model']['pretrained'])
    elif cfg['model']['base'] == 'resnet_152':
        network = Resnet_152(pretrained=cfg['model']['pretrained'])
    elif cfg['model']['base'] == 'densenet_201':
        network = Densenet_201(pretrained=cfg['model']['pretrained'])
    elif cfg['model']['base'] == 'facenet':
        network = Facenet()
    elif cfg['model']['base'] == 'ycbcrnet':
        network = Ycbcr()
    else:
        raise NotImplementedError

    return network
