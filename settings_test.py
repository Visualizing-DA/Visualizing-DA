base_architecture = 'resnet50'
img_size = 224
num_classes = 65
prototype_shape = (num_classes*10, 128, 1, 1)

prototype_activation_function = 'log'


add_on_layers_type = 'regular'
bottleneck_dim = 256

class_specific = True
init_weights = True 


source=['Ar']
target=['Pr']

import_dir = "dann/"+source[0]+"2"+target[0]
experiment_run = source[0]+'2'+target[0]+'_dann'

train_batch_size = 80*5
test_batch_size = 80*5
train_push_batch_size = 80*5

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'discriminator': 3e-3}

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,   
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    # 'l1': 0,
    # 'fts_ent': 0,
    # 'discri': 0,
    'align': 100.0,
}


data = "OfficeHome"

root = "/data/office-home"

num_train_epochs = 101
num_warm_epochs = 10

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if (i % 10 == 0)]


save_epochs = [i for i in range(num_train_epochs) if (i % 20 == 0)]
save_acc = 0.50