# Procedure control
LOAD_PRETRAIN = True
#PRETRAIN_MODEL = 'pretrain_weight_vgg16_split_2.pt'
PRETRAIN_MODEL = 'pretrain_weight_resnet18_split_cifar100.pt'
WRITE_LOG = True
TRAIN_ORI = True
VAL_ATTACK = True

# Training control
BATCH_SIZE = 100
NUM_WORKERS = 1
LR = 0.0001
EPOCHS = 40
LR_DECAY = True
TRAIN_SPLIT = True
TRAIN_IDX_S = 40000
TRAIN_IDX_E = 50000

# Attack control
ADV_LR = 1
ITERATIONS = 100
DECAY_ITER = 600
ATTACK_BATCH = 1
ATTACK_CLAMP = 0.1
ATTACK_DECAY = 0.01
ATTACK_SPLIT = True
ATTACK_IDX_S = TRAIN_IDX_S
ATTACK_IDX_E = TRAIN_IDX_E

# Validation control
SHUFFLE_VAL = True

# Pretrain control
PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 0.01
PRETRAIN_PATIENCE = 10
PRETRAIN_MODE = 'naive'
PRETRAIN_SPLIT = True
PRETRAIN_IDX_S = 0
PRETRAIN_IDX_E = 40000
PRETRAIN_CIFAR100 = True
#PRETRAIN_MODE = 'patience'
