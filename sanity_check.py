from utils import *
from parameters import *


generate_rand = True
pictures_to_show = 5


def generate_rand_noise(images_ori, mean=0, std=0):
    noise = torch.zeros_like(images_ori).normal_(mean, std)

    return images_ori + noise


def main():
    attack_split = ATTACK_SPLIT
    attack_idx_s = ATTACK_IDX_S
    attack_idx_e = ATTACK_IDX_E
    file_name = 'adv.pt'
    images_adv = torch.load(file_name)
    trainloader, testloader = cifar_dataset(batch_size=100, num_workers=1, cifar100=False)
    images_ori = torch.Tensor()
    if use_cuda:
        images_adv = images_adv.cuda()
        images_ori = images_ori.cuda()
    for idx, (inputs, labels) in enumerate(trainloader):
        if attack_split:
            if idx == 0: batch_size = len(inputs)
            idx = idx * batch_size
            if idx < attack_idx_s or idx >= attack_idx_e: continue
        if use_cuda:
            inputs = inputs.cuda()
        images_ori = torch.cat((images_ori, inputs), dim=0)
    print(f'Original stat:\n'
          f'Mean - {images_ori.mean().item()}, Max: {images_ori.max().item()}, Min: {images_ori.min().item()}')
    print(f'Adversarial stat:\n'
          f'Mean - {images_adv.mean().item()}, Max: {images_adv.max().item()}, Min: {images_adv.min().item()}')
    distance = images_adv - images_ori
    distance_mean = distance.mean()
    print(f'Mean distance is {distance_mean.item()}.')
    distance_max = distance.abs().max()
    print(f'Max distance is {distance_max.item()}')
    distance_std = distance.std()
    print(f'Standard deviation of distance is {distance_std.item()}')
    for i in range(3):
        if i == 0:
            distance_norm = distance.norm(p=1, dim=-1)
        else:
            distance_norm = distance_norm.norm(p=1, dim=-1)
    if use_cuda:
        images_adv = images_adv.cpu()
        images_ori = images_ori.cpu()
    distance_cp = distance_norm.clone()
    #idx_preset = [709, 2135, 5898, 3587, 5781]
    for i in range(pictures_to_show):
        idx = distance_cp.argmax().item()
        distance_cp[idx] = 0
        #idx = idx_preset[i]
        visualize(images_adv, image_idx=idx, filename=f'img_adv_{i}.png')
        visualize(images_ori, image_idx=idx, filename=f'img_ori_{i}.png')
        print(f'Most changed image has a distance of {distance_norm[idx].item()}, index is {idx}')
    if generate_rand:
        images_rand = generate_rand_noise(images_ori, distance_mean.item(), distance_std.item())
        rand_name = 'images_rand.pt'
        torch.save(images_rand, rand_name)
        print(f'Saved images with random perturbation as {rand_name}')


if __name__ == '__main__':
    main()
