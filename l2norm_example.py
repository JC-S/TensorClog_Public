import torch
from scipy.misc import imread, imsave


def main():
    image = 'img_ori.png'
    image_ori = torch.from_numpy(imread(image))
    image_adv = image_ori.clone()
    for i0 in range(image_ori.shape[0]):
        for i1 in range(image_ori.shape[1]):
            for i2 in range(image_ori.shape[2]):
                if i0 in range(72, 152) and i1 in range(72, 152):
                    image_adv[i0, i1, i2] = 0

    imsave('img_crop.png', image_adv.numpy())

    distance = (image_adv - image_ori).double().norm(2)
    deviant = distance**2 / image_ori.shape[0] / image_ori.shape[1] / image_ori.shape[2]
    deviant = deviant.sqrt()
    image_norm = image_ori.double() + deviant.item()
    image_norm = image_norm.clamp(0,255)
    imsave('img_norm.png', image_norm.numpy())
    image_onec = image_ori.clone()
    image_onec[:,:,0:1] = image_ori.double()[:,:,0:1] + deviant.item()*3
    image_onec = image_onec.clamp(0,255)
    imsave('img_onec.png', image_onec.numpy())


if __name__ == '__main__':
    main()
