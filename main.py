import os
import shutil
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
from proto_join import join_prototypes, join_prototypes_by_activations
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from tensorboardX import SummaryWriter as SW


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, proto_percnetile, \
    prototype_activation_function, add_on_layers_type, experiment_run, tensorboard_path

sw = SW(tensorboard_path)
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
    train_batch_size, test_batch_size, train_push_batch_size, trained_model_path, share, l2_prune

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=0, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
if trained_model_path == '':

    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type,
                                  )
else:
    ppnet = torch.load(trained_model_path)
if prototype_activation_function == 'linear':
    ppnet.set_last_layer_incorrect_connection(incorrect_strength=-1)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size, anneal_lr, reset_optim
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

from settings import last_layer_optimizer_lr, trained_optim_path
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

if trained_optim_path != '':
    trained_optims = torch.load(trained_optim_path)
    joint_optimizer.load_state_dict(trained_optims['joint_optimizer'])
    warm_optimizer.load_state_dict(trained_optims['warm_optimizer'])
    last_layer_optimizer.load_state_dict(trained_optims['last_layer_optimizer'])
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, start_epoch

# train the model
log('start training')
import copy

accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                class_specific=class_specific, log=log, sw=sw, epoch=0)
prune_not_done = False
for epoch in range(start_epoch, num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, sw=sw, epoch=epoch+1)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, sw=sw, epoch=epoch+1)
        joint_lr_scheduler.step()

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log, sw=sw, epoch=epoch+1)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch+1) + 'nopush', accu=accu,
                                target_accu=0.70, log=log, joint_optimizer=joint_optimizer, warm_optimizer=warm_optimizer,
                                last_layer_optimizer=last_layer_optimizer)

    if epoch >= push_start and epoch in push_epochs:

        push.push_prototypes(
           train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
           prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
           class_specific=class_specific,
           preprocess_input_function=preprocess_input_function, # normalize if needed
           prototype_layer_stride=1,
           root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
           epoch_number=epoch+1, # if not provided, prototypes saved previously will be overwritten
           prototype_img_filename_prefix=prototype_img_filename_prefix,
           prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
           proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
           save_prototype_class_identity=True,
           log=log,
           sw=sw,
           epoch=epoch+1,
        )
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                       class_specific=class_specific, log=log, sw=sw, epoch=epoch+1)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch+1) + 'push', accu=accu,
                                   target_accu=0.70, log=log)
        if classic_prune:
           prune_info = prune.prune_prototypes(dataloader=train_push_loader, prototype_network_parallel=ppnet_multi, k=6,
                                               prune_threshold=3, preprocess_input_function=preprocess_input_function,
                                               original_model_dir=model_dir, epoch_number=epoch, copy_prototype_imgs=False,
                                               log=print)

        if share:
            if l2_prune:
                join_info = join_prototypes(ppnet_multi, proto_percnetile, joint_optimizer, warm_optimizer, last_layer_optimizer)
            else:
                join_info = join_prototypes_by_activations(ppnet_multi, proto_percnetile, train_push_loader, joint_optimizer, warm_optimizer, last_layer_optimizer, no_p=200)
            #        np.save(os.path.join(model_dir, 'joined_prototypes_k{}'.format(proto_percnetile), 'join_info_epoch_{}.npy'.format(epoch)),
            #                np.asarray(join_info))


        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            best_acc = 0
            for i in range(25):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log, sw=sw, epoch=epoch+1)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log, sw=sw, epoch=epoch+1)
                if accu > best_acc:
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch+1) + '_' + 'push', accu=accu,
                                                target_accu=0.70, log=log, joint_optimizer=joint_optimizer, warm_optimizer=warm_optimizer,
                                                last_layer_optimizer=last_layer_optimizer)
                    best_acc = accu

        if reset_optim:
            joint_optimizer_specs = \
                [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
                 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                 ]

        if anneal_lr:
            for p in joint_optimizer.param_groups:
                p['lr'] *= 10
            joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)


logclose()

