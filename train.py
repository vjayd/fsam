import os
import torch
from torchvision import transforms, datasets
from trainer.trainer_non import Trainer_Resnet
from torch.utils.tensorboard import SummaryWriter
from models.loss import PixWiseBCELoss, Ycbcrloss
from datasets.PixWiseDataset import PixWiseDataset
from datasets.ResnetDataset import ResnetDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:0")
print(device)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

#loss = PixWiseBCELoss(beta=cfg['train']['loss']['beta'])
loss = Ycbcrloss()

writer = SummaryWriter(cfg['log_dir'])

# dump_input = torch.randn(1,3,224,224)

# writer.add_graph(network, (dump_input, ))

# Without Resize transform, images are of different sizes and it causes an error
train_transform = transforms.Compose([
    #transforms.Resize(cfg['model']['image_size']),
   # transforms.RandomRotation(cfg['dataset']['augmentation']['rotation']),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

test_transform = transforms.Compose([
   # transforms.Resize(cfg['model']['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

trainset = ResnetDataset(
    root_dir=cfg['dataset']['train_data'],
    csv_file=cfg['dataset']['train_set'],
    map_size=cfg['model']['map_size'],
    transform=train_transform,
    smoothing=cfg['model']['smoothing']
)

testset = ResnetDataset(
    root_dir=cfg['dataset']['test_data'],
    csv_file=cfg['dataset']['test_set'],
    map_size=cfg['model']['map_size'],
    transform=test_transform,
    smoothing=cfg['model']['smoothing']
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=8
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['test']['batch_size'],
    shuffle=True,
    num_workers=8
)

trainer = Trainer_Resnet(
    cfg=cfg,
    network=network,
    optimizer=optimizer,
    loss=loss,
    lr_scheduler=None,
    device=device,
    trainloader=trainloader,
    testloader=testloader,
   writer=writer
)

trainer.train()
writer.close()