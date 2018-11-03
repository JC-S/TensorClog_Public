import os
from utils import *
from parameters import *


def visualize_ori(path='./images', train=True, cifar100=False, split=False, idx_s=0, idx_e=50000):
    trainloader, testloader = cifar_dataset(cifar100=cifar100)
    if train:
        dataloader = trainloader
    else:
        dataloader = testloader
    images_ori, _ = dataloader_to_tensor(dataloader)
    pbar = tqdm(tensor_to_generator(images_ori), total=len(images_ori))
    for i, image in enumerate(pbar):
        if split:
            if i < idx_s or i >= idx_e:
                continue
            else:
                i -= idx_s
        filename = path + f'/{i}.png'
        visualize(image, image_idx=0, filename=filename)
        pbar.set_description(f'Saved {i+1} original files out of {len(images_ori)} files.')


def visualize_adv(path='./images', images_adv=torch.Tensor()):
    pbar = tqdm(tensor_to_generator(images_adv), total=len(images_adv))
    for i, image in enumerate(pbar):
        filename = path + f'/{i}.png'
        visualize(image, image_idx=0, filename=filename)
        pbar.set_description(f'Saved {i+1} adversarial files out of {len(images_adv)} files.')


def main():
    path = './images_ori'
    if not os.path.exists(path):
        os.makedirs(path)
    print('Visualizing original images...')
    visualize_ori(path=path, train=True, cifar100=False,
                  split=TRAIN_SPLIT, idx_s=TRAIN_IDX_S, idx_e=TRAIN_IDX_E)
    print('-'*140)
    path = './images_adv'
    if not os.path.exists(path):
        os.makedirs(path)
    print('Visualizing adversarial images...')
    adv_file = 'adv.pt'
    images_adv = torch.load(adv_file)
    visualize_adv(path=path, images_adv=images_adv)
    print('-'*140)


if __name__ == '__main__':
    main()
