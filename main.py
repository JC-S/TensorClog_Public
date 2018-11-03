import os

from attacks import attack_iter
from models import *
from utils import *
from parameters import *


def get_model():
    #model = VGG('VGG16')
    model = ResNet18(num_classes=100)

    return model


def main():
    load_pretrain = LOAD_PRETRAIN
    pretrain_model = PRETRAIN_MODEL
    write_log = WRITE_LOG
    train_ori = TRAIN_ORI
    val_attack = VAL_ATTACK
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    lr = LR
    epochs = EPOCHS
    lr_decay = LR_DECAY
    train_split = TRAIN_SPLIT
    train_idx_s = TRAIN_IDX_S
    train_idx_e = TRAIN_IDX_E
    adv_lr = ADV_LR
    iterations = ITERATIONS
    decay_iter = DECAY_ITER
    attack_batch = ATTACK_BATCH
    attack_clamp = ATTACK_CLAMP
    attack_decay = ATTACK_DECAY
    attack_split = ATTACK_SPLIT
    attack_idx_s = ATTACK_IDX_S
    attack_idx_e = ATTACK_IDX_E
    shuffle_val = SHUFFLE_VAL
    trainloader, testloader = cifar_dataset(batch_size, num_workers, shuffle_train=False, cifar100=False)
    criterion = nn.CrossEntropyLoss()
    model = get_model()
    print('Loading pretrained model.')
    if load_pretrain:
        model.load_state_dict(torch.load(pretrain_model))
    transfer(model, class_num=100)
    if use_cuda:
        model = model.cuda()
    print('Training network...')
    stat_error_train_ori = []
    stat_error_test_ori = []
    stat_loss_ori = []
    if train_ori:
        for epoch in range(epochs):
            if lr_decay:
                if epoch == 0:
                    print(f'Set lr to {lr:.9f}')
                if epoch % (epochs/2) == 0 and epoch != 0:
                    lr *= 0.1
                    print(f'Set lr to {lr:.9f}')
                #if epoch == epochs / 2:
                #    lr = LR
                #    print(f'Set lr to {lr:.9f}')
            # if epoch < epochs / 2:
            #     optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr*10, momentum=0.9)
            # else:
            #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr*10, momentum=0.9)
            error_train, error_test, loss = train(model, criterion, trainloader, testloader, optimizer,
                                                  1, epoch_count=epoch+1,
                                                  split=train_split, idx_s=train_idx_s, idx_e=train_idx_e)
            stat_error_train_ori += [error_train.item()]
            stat_error_test_ori += [error_test.item()]
            stat_loss_ori += [loss.item()]
        print('-'*140)

    save_name = ['adv.pt', 'labels.pt']
    if os.path.exists(save_name[0]) and os.path.exists(save_name[1]):
        print('Adversarial file already exists.')
        train_adv = torch.load(save_name[0])
        label_adv = torch.load(save_name[1])
        print('-'*140)
    else:
        print('Generating attack...')
        model = get_model()
        model.load_state_dict(torch.load(pretrain_model))
        transfer(model, class_num=100)
        if use_cuda:
            model = model.cuda()
        trainloader, testloader = cifar_dataset(attack_batch, num_workers=0, shuffle_train=False, cifar100=False)
        train_adv, label_adv = attack_iter(model, criterion, trainloader, adv_lr,
                                           save_name=save_name, save_adv=True,
                                           iterations=iterations, decay_iter=decay_iter,
                                           attack_clamp=attack_clamp, attack_decay=attack_decay,
                                           split=attack_split, idx_s=attack_idx_s, idx_e=attack_idx_e)
        print('-'*140)
    trainset_adv = torch.utils.data.TensorDataset(train_adv, label_adv)
    trainloader_adv = torch.utils.data.DataLoader(trainset_adv, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_val)
    model = get_model()
    if load_pretrain:
        model.load_state_dict(torch.load(pretrain_model))
    # torch.manual_seed(466)
    # torch.cuda.manual_seed_all(466)
    transfer(model, class_num=100)
    if use_cuda:
        model = model.cuda()
    print('Validating attack...')
    stat_error_train_adv = []
    stat_error_test_adv = []
    stat_loss_adv = []
    lr = LR
    if val_attack:
        for epoch in range(epochs):
            if lr_decay:
                if epoch == 0:
                    print(f'Set lr to {lr:.9f}')
                if epoch % (epochs/2) == 0 and epoch != 0:
                    lr *= 0.1
                    print(f'Set lr to {lr:.9f}')
                #if epoch == epochs / 2:
                #    lr = LR
                #    print(f'Set lr to {lr:.9f}')
            # if epoch < epochs / 2:
            #     optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr*10, momentum=0.9)
            # else:
            #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr*10, momentum=0.9)
            error_train, error_test, loss = train(model, criterion, trainloader_adv, testloader, optimizer,
                                                  1, epoch_count=epoch+1)
            stat_error_train_adv += [error_train.item()]
            stat_error_test_adv += [error_test.item()]
            stat_loss_adv += [loss.item()]
        print('-'*140)
    if write_log:
        print('Writing csv log...')
        if train_ori:
            write_csv(stat_error_train_ori, stat_error_test_ori, stat_loss_ori, 'stat_ori.csv')
        if val_attack:
            write_csv(stat_error_train_adv, stat_error_test_adv, stat_loss_adv, 'stat_adv.csv')
        print('-'*140)


if __name__ == '__main__':
    main()