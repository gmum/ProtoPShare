import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop
from torchvision.transforms import ToTensor
import os
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from proto_join import join_prototypes
import model

train_push_dataset = datasets.ImageFolder(
    '../dataset/train_birds/',
    # '/Users/bartoszzielinski/Databases/birds/train_birds/',
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ]))


train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=80, shuffle=False,
    num_workers=0, pin_memory=False)

ppnet = model.construct_PPNet(base_architecture='vgg19',
                              pretrained=True, img_size=224,
                              prototype_shape=(2000, 128, 1, 1),
                              num_classes=200,
                              prototype_activation_function='log',
                              add_on_layers_type='regular')

ppnet = torch.load('saved_models/vgg19/003/30nopush0.7323.pth')
# ppnet = torch.load('./pretrained_models/30nopush0.7323.pth', map_location=torch.device('cpu'))
ppnet_multi = torch.nn.DataParallel(ppnet)

from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

from proto_join import join_prototypes

join_prototypes(ppnet_multi, 0.1, joint_optimizer, warm_optimizer, last_layer_optimizer)

print('Prototypes after joining')
print(ppnet_multi.module.prototype_vectors.data.shape)
