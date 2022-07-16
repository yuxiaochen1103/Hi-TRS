import torch
from dataset.NTU_3D_AR import NTU_Dataset, my_collate
import yaml
import numpy as np
from model.Hi_TRS_AR import Hi_TRS_AR
import os
from tqdm import tqdm


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_data_loader(ckpt_cfg, ft_cfg):

    #try:
    test_dataset = NTU_Dataset(benchmark=ft_cfg.benchmark, data_split='val',
                               w_size=ft_cfg.w_size, stride=ft_cfg.stride, dilate=ft_cfg.dilate,
                               use_data_aug=False, view=ckpt_cfg.view,
                               label_num=ft_cfg.label_num)


    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=ft_cfg.workers,
        collate_fn=my_collate,
        pin_memory=True,
        shuffle=False,
    )

    return val_loader

def init_model(ckpt_cfg):

    model = Hi_TRS_AR(
                    w_size=ckpt_cfg.w_size,
                    d_model=ckpt_cfg.d_model,
                    nhead=ckpt_cfg.nhead, d_att=ckpt_cfg.d_att,
                    layer_num=ckpt_cfg.layer_num,
                    dp_rate=ckpt_cfg.dp_rate,
                    label_num=ft_cfg.label_num
                      )
    model = torch.nn.DataParallel(model).cuda().float()

    return model

def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def val():

    #***********evaluation***********
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader)):

            joint_seq = batch['joint_seq'].cuda().float()
            label = batch['label'].cuda().long()
            w_pad_mask = batch['w_pad_mask'].cuda().long()

            score = model(joint_seq, w_pad_mask=w_pad_mask)

            #record score and label list
            if i == 0:
                score_list = score.data.cpu()
                label_list = label.data.cpu()
            else:
                score_list = torch.cat((score_list, score.data.cpu()), 0)
                label_list = torch.cat((label_list, label.data.cpu()), 0)


        val_acc = get_acc(score_list,label_list)
        print('testing acc:', val_acc)
        score_list = score_list.data.cpu().numpy()
        label_list = label_list.data.cpu().numpy()
    return score_list, label_list


if __name__ == "__main__":


    ckpt_fold = './training_log/AR/NTU_60/xsub/ft_all/ckpt-iter-num-30000_part-all_view-joint'
    ckpt_name = 'epoch-50.pth'

    ckpt_cfg_pth = ckpt_fold + '/ckpt_cfg.yaml'
    ft_cfg_pth = ckpt_fold + '/ft_cfg.yaml'


    #load model config information
    cfg_file = open(ckpt_cfg_pth)
    ckpt_cfg = yaml.load(cfg_file)
    ckpt_cfg = AttrDict(ckpt_cfg)

    #load fine_tuneing config information
    cfg_file = open(ft_cfg_pth)
    ft_cfg = yaml.load(cfg_file)
    ft_cfg = AttrDict(ft_cfg)

    print(ckpt_cfg)
    print(ft_cfg)


    #.... get dataloader
    val_loader = get_data_loader(ckpt_cfg, ft_cfg)


    #.........inital model
    print("\ninit model.............")
    model = init_model(ckpt_cfg)


    print('loading weight......')

    ckpt_pth = os.path.join(ckpt_fold, 'model', ckpt_name)
    print(ckpt_pth)
    pretrained_dict = torch.load(ckpt_pth)
    model_dict = model.state_dict()
    model.load_state_dict(pretrained_dict)

    #parameters recording training lo
    prd_score, label_list = val()

    print(prd_score.shape, label_list.shape)

    assert prd_score.shape[0] == label_list.shape[0]

    np.savez(ckpt_fold + '/test_result.npz', prd=prd_score, label=label_list)