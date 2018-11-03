import json
import pandas as pd
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from furniture_params import *
from attacks import attack_iter
from utils import *

import models_furniture as models

import os


class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        path = f'data/{path}.json'
        self.transform = transform
        img_idx = {int(p.name.split('.')[0])
                   for p in Path(f'data/{preffix}').glob('*.jpg')}
        data = json.load(open(path))
        if 'annotations' in data:
            data = pd.DataFrame(data['annotations'])
        else:
            data = pd.DataFrame(data['images'])
        self.full_data = data
        nb_total = data.shape[0]
        data = data[data.image_id.isin(img_idx)].copy()
        data['path'] = data.image_id.map(lambda i: f"data/{preffix}/{i}.jpg")
        self.data = data
        print(f'[+] dataset `{preffix}` loaded {data.shape[0]} images from {nb_total}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)
        target = row['label_id'] - 1 if 'label_id' in row else -1
        return img, target


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])


use_gpu = use_cuda


def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.resnet152_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def main():
    trainset = FurnitureDataset('train', transform=preprocess)
    trainloader = DataLoader(dataset=trainset, num_workers=NB_WORKERS, batch_size=1, shuffle=False)
    if not os.path.exists(ATTACK_SAVE_PATH):
        os.makedirs(ATTACK_SAVE_PATH)
    model = get_model()
    criterion = torch.nn.CrossEntropyLoss()
    #transfer(model, class_num=NB_CLASSES)
    attack_iter(model, criterion, trainloader,
                adv_lr=ADV_LR, transfer=True,
                iterations=ITERATIONS,
                instant_save=True, save_path=ATTACK_SAVE_PATH,
                attack_clamp=ATTACK_CLAMP, attack_decay=ATTACK_DECAY,
                furniture=True,
                split=ATTACK_SPLIT, idx_s=ATTACK_IDX_S, idx_e=ATTACK_IDX_E)


if __name__ == '__main__':
    main()