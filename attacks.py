import torch
from torch.autograd import grad
from tqdm import tqdm
from scipy.misc import imsave

from utils import use_cuda


def attack_iter(model, criterion, dataloader, adv_lr, save_name=['adv.pt', 'labels.pt'], transfer=True, save_adv=False,
                iterations=3,
                decay_iter=1,
                instant_save=False, save_path=None,
                attack_clamp = False, attack_decay = False,
                split=False, idx_s=0, idx_e=50000,
                furniture=False):
    model.eval()
    inputs_adv_all = torch.Tensor()
    inputs_labels = torch.Tensor().type(torch.LongTensor)
    norm_ori = torch.Tensor()
    norm_adv = torch.Tensor()
    reduced_percent = torch.Tensor()
    if use_cuda:
        inputs_adv_all = inputs_adv_all.cuda()
        inputs_labels = inputs_labels.cuda()
        norm_ori = norm_ori.cuda()
        norm_adv = norm_adv.cuda()
        reduced_percent = reduced_percent.cuda()
    pbar = tqdm(dataloader, total=len(dataloader))
    if attack_clamp:
        import numpy as np
    for idx, (inputs, labels) in enumerate(pbar):
        if split:
            if idx == 0: batch_size = len(inputs)
            idx = idx * batch_size
            if idx < idx_s or idx >= idx_e: continue
        inputs.requires_grad = True
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs_ori = inputs.clone()
        if attack_clamp:
            if use_cuda:
                inputs_max = (inputs_ori + attack_clamp).cpu().data.numpy()
                inputs_min = (inputs_ori - attack_clamp).cpu().data.numpy()
            else:
                inputs_max = (inputs_ori + attack_clamp).data.numpy()
                inputs_min = (inputs_ori - attack_clamp).data.numpy()
        inputs_adv = inputs
        adv_lr_c = adv_lr
        pbar_l = tqdm(range(iterations))
        for i in pbar_l:
            if i % decay_iter == 0 and i != 0:
                adv_lr_c *= 0.5
            W_grad_l2norm = torch.zeros(1)
            W_grad_l2norm.requires_grad = True
            if use_cuda:
                W_grad_l2norm = W_grad_l2norm.cuda()
            outputs = model(inputs_adv)
            loss = criterion(outputs, labels)
            if transfer:
                if furniture:
                    attack_layer = model.fresh_params()
                else:
                    attack_layer = model.classifier.parameters()
            else:
                attack_layer = model.parameters()
            for w in attack_layer:
                dloss_dw, = grad(loss, w, create_graph=True)
                W_grad_l2norm += dloss_dw.norm(2)
            # Keep original W_l2norm for evaluation
            if i == 0:
                W_grad_l2norm_ori = W_grad_l2norm.data
            # L_T as TensorClog loss.
            if attack_decay:
                regularizer = (inputs_ori.data - inputs_adv).norm(2)
                L_T = W_grad_l2norm + attack_decay*regularizer
            else:
                L_T = W_grad_l2norm
            dWl2_dinputs, = grad(L_T, inputs_adv)
            inputs_adv = inputs_adv - dWl2_dinputs*adv_lr_c
            if attack_clamp:
                if use_cuda:
                    inputs_adv_np = inputs_adv.data.cpu().numpy()
                else:
                    inputs_adv_np = inputs_adv.data.numpy()
                inputs_adv_np = np.clip(inputs_adv_np, a_min=inputs_min, a_max=inputs_max)
                inputs_adv = torch.from_numpy(inputs_adv_np)
                if use_cuda:
                    inputs_adv = inputs_adv.cuda()
                inputs_adv.requires_grad = True
            pbar_l.set_description(f'Iter: {i} - W_l2norm: {W_grad_l2norm.item():5f} - L_T loss: {L_T.item():5f}.')
        if instant_save:
            if split:
                idx_save = idx + 1
            inputs_adv_np = np.rollaxis(inputs_adv_np, 1, 4)
            imsave(f'{save_path}/{idx_save}.png', inputs_adv_np[0])
        else:
            inputs_adv_all = torch.cat((inputs_adv_all, inputs_adv.data), dim=0)
            inputs_labels = torch.cat((inputs_labels, labels.data), dim=0)
        norm_ori = torch.cat((norm_ori, W_grad_l2norm_ori.data), dim=0)
        norm_adv = torch.cat((norm_adv, W_grad_l2norm.data), dim=0)
        pbar.set_description(f'Attack - norm: {W_grad_l2norm_ori.item():5f} - reduce: {(W_grad_l2norm_ori-W_grad_l2norm).item():5f}')
        if not instant_save:
            reduced = (W_grad_l2norm_ori-W_grad_l2norm)/W_grad_l2norm_ori
            reduced_percent = torch.cat((reduced_percent, reduced), dim=0)
    norm_ori = norm_ori.mean()
    norm_adv = norm_adv.mean()
    print(f'Norm ori mean: {norm_ori:5f}, Norm adv mean: {norm_adv:5f}')
    if not instant_save:
        print(f'Mean reduced percent: {reduced_percent.mean().item():5f}')
    if use_cuda:
        inputs_adv_all = inputs_adv_all.cpu()
        inputs_labels = inputs_labels.cpu()
    if save_adv and not instant_save:
        torch.save(inputs_adv_all, save_name[0])
        torch.save(inputs_labels, save_name[1])
    return inputs_adv_all, inputs_labels
