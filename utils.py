import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import csv
from tqdm import tqdm
from scipy.misc import imsave


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(233)
torch.cuda.manual_seed_all(233)


def cifar_dataset(batch_size=128, num_workers=2, download=False, shuffle_train=False, cifar100=False):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if cifar100:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)

    if cifar100:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=download, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def init_xavier_uniform(model):
    for w in model.parameters():
        if len(w.shape) > 1:
            nn.init.xavier_uniform_(w)

    return model


def predict(model, testloader, split=False, idx_s=0, idx_e=50000):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(testloader):
            if split:
                if batch_id == 0:
                    batch_size = len(inputs)
                idx = batch_id * batch_size
                if idx < idx_s or idx >= idx_e: continue
            all_labels.append(labels)
            if use_cuda:
                inputs = inputs.cuda()

            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())

        all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_cuda:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


def train(model, criterion, trainloader, testloader, optimizer, epochs, epoch_count=None, split=False, idx_s=0, idx_e=50000):
    model.train()
    for epoch in range(epochs):
        epoch += 1
        pbar = tqdm(trainloader, total=len(trainloader))
        train_loss_all = .0
        epoch_print = epoch if epoch_count is None else epoch_count
        for batch_id, (inputs, labels) in enumerate(pbar):
            if split:
                if batch_id == 0:
                    skip = 0
                    batch_size = len(inputs)
                idx = batch_id * batch_size
                if idx < idx_s or idx >= idx_e: skip+=1; continue
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_all += loss.data
            if split:
                batch_id -= skip
            train_loss = train_loss_all/(batch_id+1)
            pbar.set_description(f'Epoch: {epoch_print} - loss: {train_loss:5f}.')

        labels, outputs = predict(model, testloader)
        _, preds = torch.max(outputs.data, dim=1)
        error_test = torch.mean((preds!=labels.data).float()).data
        labels, outputs = predict(model, trainloader, split=split, idx_s=idx_s, idx_e=idx_e)
        _, preds = torch.max(outputs.data, dim=1)
        error_train = torch.mean((preds!=labels.data).float()).data
        print(f'train_acc: {1-error_train:5f} - val_acc: {1-error_test:5f}')

    return error_train, error_test, train_loss


def transfer(model, class_num, randseed=233, init=nn.init.xavier_uniform_):
    fan_in = next(model.linear.parameters()).shape[-1]
    model.linear = nn.Linear(fan_in, class_num)
    torch.manual_seed(randseed)
    init(next(model.linear.parameters()))


def write_csv(error_train, error_test, loss, filename):
    if not len(error_train) == len(error_test) == len(loss):
        raise ValueError('Length of inputs do not match.')
    else:
        epoch = range(len(error_train))
    csvfile = open(filename, 'w')
    fieldnames = ['epoch', 'error_train', 'error_test', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(error_train)):
        writer.writerow({'epoch': epoch[i],
                         'error_train': error_train[i],
                         'error_test': error_test[i],
                         'loss': loss[i]})


def tensor_to_generator(tensor):
    i = 0
    while i < len(tensor):
        yield tensor[i:i+1]
        i += 1


def dataloader_to_tensor(dataloader, get_label=False):
    images = torch.Tensor()
    labels = torch.Tensor()
    if use_cuda:
        images = images.cuda()
        labels = labels.cuda()
    for inputs, targets in dataloader:
        if use_cuda:
            inputs = inputs.cuda()
            if get_label:
                targets = targets.cuda()
        images = torch.cat((images, inputs), dim=0)
        if get_label:
            labels = torch.cat((labels, targets), dim=0)
    if use_cuda:
        images = images.cpu()
        labels = labels.cpu()

    return images, labels


def visualize(imagetensor, image_idx=1, filename='/tmp/tmp.png'):
    images = imagetensor.permute(0,2,3,1).numpy()
    imsave(filename, images[image_idx], format='png')
