import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir

import model_test as model
import push_reload as push
import prune
import train_and_test_reload_DANN_Cls as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

from tllib.utils.data import ForeverDataIterator
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3

parser.add_argument('--train-resizing', type=str, default='default')
parser.add_argument('--val-resizing', type=str, default='default')

parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings_test import base_architecture, img_size, prototype_shape, num_classes, init_weights,\
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_test.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model_test.py'), dst=model_dir)


shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test_reload_DANN_Cls.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
# from settings_test import train_dir, test_dir, train_push_dir, \
#                      train_batch_size, test_batch_size, train_push_batch_size, bottleneck_dim, class_specific
                     
from settings_test import data, root, source, target, img_size, import_dir, \
                train_batch_size, test_batch_size, train_push_batch_size, bottleneck_dim, class_specific
                # train_dir, test_dir, train_push_dir,\

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
#                                                 random_horizontal_flip=not args.no_hflip,
#                                                 random_color_jitter=False, resize_size=args.resize_size,
#                                                 norm_mean=args.norm_mean, norm_std=args.norm_std)
push_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ])
# test_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
#                                             norm_mean=args.norm_mean, norm_std=args.norm_std)

train_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

test_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
# push_transform = transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#     ])
# all datasets
# train set

train_source_dataset, train_target_dataset, train_push_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_train_push_test(data, root, source, target, train_transform, push_transform, test_transform)


train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)


train_source_loader = torch.utils.data.DataLoader(
    train_source_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False, drop_last=True)
train_target_loader = torch.utils.data.DataLoader(
    train_target_dataset, batch_size=test_batch_size, shuffle=True,
    num_workers=4, pin_memory=False, drop_last=True)
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_source_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)

num_iters = max(len(train_source_loader), len(train_target_loader))
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              init_weights=init_weights, 
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              bottleneck_dim=bottleneck_dim)

backbone = utils.get_model('resnet34', pretrain=True)
checkpoint = torch.load("./"+ import_dir +"/Ar2Cl.pth")
pool_layer = None
classifier = ImageClassifier(backbone, 65, bottleneck_dim=256,
                                 pool_layer=pool_layer, finetune=False)

classifier.load_state_dict(checkpoint)
ppnet.features.conv1.load_state_dict(classifier.backbone.conv1.state_dict())
ppnet.features.bn1.load_state_dict(classifier.backbone.bn1.state_dict())
ppnet.features.relu.load_state_dict(classifier.backbone.relu.state_dict())
ppnet.features.maxpool.load_state_dict(classifier.backbone.maxpool.state_dict())
ppnet.features.layer1.load_state_dict(classifier.backbone.layer1.state_dict())
ppnet.features.layer2.load_state_dict(classifier.backbone.layer2.state_dict())
ppnet.features.layer3.load_state_dict(classifier.backbone.layer3.state_dict())
ppnet.features.layer4.load_state_dict(classifier.backbone.layer4.state_dict())
ppnet.bottleneck.load_state_dict(classifier.bottleneck.state_dict())
ppnet.fc.load_state_dict(classifier.head.state_dict())

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
# class_specific = False


domain_discri = DomainDiscriminator(in_feature=bottleneck_dim, hidden_size=1024).cuda()
domain_adv = DomainAdversarialLoss(domain_discri).cuda()

# define optimizer
from settings_test import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': domain_discri.parameters(), 'lr': joint_optimizer_lrs['discriminator']},
]
# print("features", ppnet.features.parameters())

# print("domain_discri", domain_discri.parameters())


joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings_test import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings_test import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings_test import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings_test import num_train_epochs, num_warm_epochs, push_start, push_epochs, save_acc

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, domain_discri=domain_discri, domain_adv=domain_adv, dataloader=train_source_iter, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, target_loader=train_target_iter, num_iters=num_iters)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        
        _ = tnt.train(model=ppnet_multi, domain_discri=domain_discri, domain_adv=domain_adv, dataloader=train_source_iter, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, target_loader=train_target_iter, num_iters=num_iters)
        joint_lr_scheduler.step()

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    if (epoch%10==9) or (epoch%10==1) or (epoch%10==0):
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=save_acc, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        if (epoch%10==9) or (epoch%10==1) or (epoch%10==0):
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=save_acc, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                # _ = tnt.train_last_only(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                #               class_specific=class_specific, coefs=coefs, log=log)
                _ = tnt.train_last_only(model=ppnet_multi, domain_discri=domain_discri, domain_adv=domain_adv, dataloader=train_source_iter, optimizer=last_layer_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, target_loader=train_target_iter, num_iters=num_iters)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                if ((epoch%10==9) or (epoch%10==1) or (epoch%10==0)) and (i%9==0):
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=save_acc, log=log)
    
    
    if epoch >= push_start and epoch in push_epochs:
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'backup', accu=accu,
                                            target_accu=0.1, log=log)
logclose()

