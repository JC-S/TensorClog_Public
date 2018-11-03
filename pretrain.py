from models import *
from utils import *
from parameters import *


def main():
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    epochs = PRETRAIN_EPOCHS
    patience_max = PRETRAIN_PATIENCE
    lr = PRETRAIN_LR
    mode = PRETRAIN_MODE
    pretrain_split = PRETRAIN_SPLIT
    pretrain_idx_s = PRETRAIN_IDX_S
    pretrain_idx_e = PRETRAIN_IDX_E
    pretrain_cifar100 = PRETRAIN_CIFAR100
    print('Building network...')
    #model = VGG('VGG16', num_classes = 100 if pretrain_cifar100 else 10)
    model = ResNet18(num_classes = 100 if pretrain_cifar100 else 10)
    trainloader, testloader = cifar_dataset(batch_size, num_workers, shuffle_train=False, cifar100=pretrain_cifar100)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print('Training network...')
    print(f'Initial lr is {lr:.5f}')
    if mode == 'patience':
        patience = 0
        loss_min = float('inf')
        for epoch in range(epochs):
            epoch += 1
            if patience >= patience_max:
                lr *= 0.5
                patience = 0
                model.load_state_dict(torch.load('pretrain_weight.pt'))
                print(f'Patience exceeded. Reset lr to {lr:.5f}')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            _, error, loss = train(model, criterion, trainloader, testloader, optimizer, 1, epoch_count=epoch)
            if loss < loss_min:
                patience = 0
                loss_min = loss
                error_min = error
                torch.save(model.state_dict(), 'pretrain_weight.pt')
            else:
                patience += 1
        print('-'*140)
        print(f'Best model stat: Loss = {loss_min:.5f}, Error ={error_min:.5f}')
    if mode == 'naive':
        best_acc = 0
        best_loss = 0
        for epoch in range(epochs):
            if epoch % 20 == 0 and epoch != 0:
                lr *= 0.5
                print(f'Set lr to {lr:.5f}')
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
            _, error, loss = train(model, criterion, trainloader, testloader, optimizer, 1, epoch_count=epoch+1,
                                   split=pretrain_split, idx_s=pretrain_idx_s, idx_e=pretrain_idx_e)
            acc = 1 - error
            if acc > best_acc:
                best_acc = acc
                best_loss = loss
                torch.save(model.state_dict(), 'pretrain_weight.pt')
        print('-'*140)
        print(f'Best model stat: Loss = {best_loss:.5f}, Error = {(1-best_acc):.5f}')


if __name__ == '__main__':
    main()