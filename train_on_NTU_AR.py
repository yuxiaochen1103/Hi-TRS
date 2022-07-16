import time
import torch
from dataset.NTU_3D_AR import NTU_Dataset, my_collate
import yaml
import torch.optim as optim
import numpy as np
from model.Hi_TRS_AR import Hi_TRS_AR
import argparse
from tensorboardX import SummaryWriter
import os
from shutil import copyfile
import random

parser = argparse.ArgumentParser()

parser.add_argument("--ft_cfg_pth")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_data_loader(ckpt_cfg, ft_cfg):
    #ckpt_cfg.view = 'joint'
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    #benchmark, data_split, w_size, stride, use_data_aug
    #try:
    train_dataset = NTU_Dataset(benchmark=ckpt_cfg.benchmark, data_split='train',
                               w_size=ckpt_cfg.w_size, stride=ckpt_cfg.stride, dilate=ckpt_cfg.dilate,
                               use_data_aug=True, view=ckpt_cfg.view, data_part=ft_cfg.data_part, label_num=ckpt_cfg.label_num)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=ft_cfg.batch_size,
        num_workers=ft_cfg.workers,
        collate_fn=my_collate,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader

def init_model(ckpt_cfg):

    #w_size, d_model, dim_feedforward, nhead, d_att, dp_rate, layer_num, use_spe, se_type
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

def train():

    if not ft_cfg.linear:
        model.train()
    else:
        print('model eval')
        model.eval()


    #use to record training status
    loss_name_list = ['loss', 'acc']
    lossname_2_id = {}
    for i, name in enumerate(loss_name_list):
        lossname_2_id[name] = i

    epoch_train_log = 0. # whole epoch
    temp_train_log = 0. #every 10 iteration

    #to record time usage
    io_time = 0
    train_time = 0
    start_time = time.time()

    model.zero_grad()
    for i, batch in enumerate(train_loader):


        #load data
        joint_seq = batch['joint_seq'].cuda().float()
        label = batch['label'].cuda().long()
        w_pad_mask = batch['w_pad_mask'].cuda().long()

        #print(joint_seq.shape)

        #calcult I/O time
        now = time.time()
        io_time += now - start_time
        start_time = now

        #forward pass

        score = model(joint_seq, w_pad_mask=w_pad_mask)

        #calc loss and acc
        loss = criterion(score, label)
        acc = get_acc(score, label)

        #backward
        loss /= ft_cfg.accum_iter
        loss.backward()

        #print(model.module.v_pe.parameter.grad)
        #print(model.module.v_pe.parameter)

        if (i + 1) % ft_cfg.accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            model_solver.step()
            model.zero_grad()
            model.module.iter_num += 1

        #record loss and acc
        epoch_train_log += np.array([loss.item(), acc])
        temp_train_log += np.array([loss.item(), acc])

        #cal model running time
        now = time.time()
        train_time += now - start_time
        start_time = now

        #print time_usage / loss / acc every 10 iteration
        if (i + 1) % ft_cfg.accum_iter == 0:
            temp_train_log /= ft_cfg.accum_iter
            print("*** Epoch: [{}] [{} of {}] IO_time: {} train_time: {} lr: {} ".format(epoch, i + 1, len(train_loader),
                                                                                  round(io_time, 3),
                                                                                  round(train_time, 3),
                                                                                 round(model_solver.param_groups[0]['lr'], 6)), end='')
            for loss_name in loss_name_list:
                loss_id = lossname_2_id[loss_name]
                print("{} {:.4f} ".format(loss_name, temp_train_log[loss_id]), end='')
            print()

            temp_train_log = 0
            io_time = 0
            train_time = 0

    #save result to tensorboard every epoch

    epoch_train_log /= (i + 1)
    for loss_name in loss_name_list:
        loss_id = lossname_2_id[loss_name]
        writer.add_scalar('Train/{}'.format(loss_name), epoch_train_log[loss_id], model.module.iter_num)
    return epoch_train_log

def get_ckpt_list(model_fold):
    ckpt_list = os.listdir(model_fold)
    #print(ckpt_list)
    ckpt_list = [ele for ele in ckpt_list if 'iter-' in ele]
    iter_2_ckpt = {}
    for ckpt in ckpt_list:
        iter = int(ckpt.split('iter-')[1].split('_train')[0])
        iter_2_ckpt[iter] = ckpt
    #print(ckpt_list[:5])
    return iter_2_ckpt

def lr_lambda_train_scratch(current_step: int):

    #no_decay_epoch = 30
    if current_step <= 30:
        return 1
    elif 30 < current_step <= 40:
        return 0.5  # as LambdaLR multiplies by lr_init -->0.0002
    elif 40 < current_step <= 50:
        return 0.5 * 0.5 #--->0.0001
    elif 50 < current_step <= 55:
        return 0.5 * 0.5 * 0.5 #--->0.00005
    elif 55 < current_step:
        return 0.5 * 0.5 * 0.5 * 0.1 #--->0.0001


def lr_lambda_ft(current_step: int):
    decay_epoch = 20
    if current_step <= decay_epoch:
        return 1
    else:
        t = (current_step - decay_epoch) / 10
        t = np.ceil(t)
        factor = max(0.5 ** t, 0.01 / 4)

        return factor

def lr_lambda_linear(current_step: int):
    decay_epoch = 100
    if current_step <= decay_epoch:
        return 1
    else:
        t = (current_step - decay_epoch) / 10
        t = np.ceil(t)
        factor = max(0.5 ** t, 0.01)

        return factor



seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":



    #*** load config
    args = parser.parse_args()

    #load fine-tune config information
    cfg_file = open(args.ft_cfg_pth)
    ft_cfg = yaml.load(cfg_file)
    ft_cfg = AttrDict(ft_cfg)

    if not ft_cfg.load_pretrained:
        ckpt_cfg = ft_cfg
        ckpt_cfg_pth = args.ft_cfg_pth
    else:
        ckpt_folder = './checkpoint/'

        print('ckpt_folder:')
        print(ckpt_folder)

        ckpt_cfg_pth = os.path.join(ckpt_folder, 'training_cfg.yaml')
        cfg_file = open(ckpt_cfg_pth)
        ckpt_cfg = yaml.load(cfg_file)
        ckpt_cfg = AttrDict(ckpt_cfg)

    #load

    print('-' * 30)
    print(args)
    print(ckpt_cfg)
    print(ft_cfg)
    print('-' * 30)
    print()


    #.... get dataloader
    train_loader = get_data_loader(ckpt_cfg, ft_cfg)


    #.........inital model
    print("\ninit model.............")
    model = init_model(ckpt_cfg)


    if ft_cfg.load_pretrained:
        print('loading weight......')

        iter_2_ckpt = get_ckpt_list(os.path.join(ckpt_folder, 'model'))
        ckpt_name = iter_2_ckpt[ft_cfg.ckpt_iter_num]

        print(ckpt_name)
        ckpt_pth = os.path.join(ckpt_folder, 'model', ckpt_name)
        pretrained_dict = torch.load(ckpt_pth)


        model_dict = model.state_dict()
        #print(len(model_dict.keys()), len(pretrained_dict.keys()))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        #print(len(model_dict.keys()), len(pretrained_dict.keys()))
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        #model.load_state_dict(state_dict)

    # model.module.v_cls = torch.nn.Sequential(
    #     torch.nn.Linear(ckpt_cfg.d_model * 4, ft_cfg.label_num)
    # ).cuda()




    if ft_cfg.linear:
        print('freezing layer...')
        for name, param in model.named_parameters():
            if 'v_cls' not in name:
                param.requires_grad = False
            else:
                print(name)
        ft_cfg.weight_decay = 0
        print('weight decay:', ft_cfg.weight_decay)


    no_decay = ["bias", "LayerNorm.weight", '_norm', 's_input_map.2.weight', '_pe.',
                'fuse_token']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": ft_cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    model_solver = optim.AdamW(
        optimizer_grouped_parameters,
        lr=ft_cfg.lr)

    #scheduler = optim.lr_scheduler.StepLR(model_solver, step_size=25, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_solver, 'max', patience=3,
    #                                                   verbose=True, min_lr=1e-6, factor=0.5)

    if ft_cfg.linear:
        lr_lambda = lr_lambda_linear
    else:
        if ft_cfg.load_pretrained:
            lr_lambda = lr_lambda_ft
        else:
            lr_lambda = lr_lambda_train_scratch

    scheduler = optim.lr_scheduler.LambdaLR(model_solver, lr_lambda=lr_lambda)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss().cuda()

    log_pth = ft_cfg.log_pth

    #....tensorboard logger
    if ft_cfg.load_pretrained:
        if ft_cfg.linear:
            setting = 'ft_linear'
        else:
            setting = 'ft_all'
        model_name = 'ckpt-iter-num-{}_part-{}_view-{}'.format(ft_cfg.ckpt_iter_num,
                                                                                            ft_cfg.data_part, ckpt_cfg.view)
    else:
        if ft_cfg.linear:
            setting = 'bsl_linear'
        else:
            setting = 'bsl_all'
        model_name = 'dim-{}_nhead-{}_d-att-{}_layer-{}_dp-rate-{}_wd-{}_lr-{}_w-{}_s-{}_d-{}_view-{}_part-{}'.format(
                                                                                ckpt_cfg.d_model, ckpt_cfg.nhead, ckpt_cfg.d_att, ckpt_cfg.layer_num,
                                                                                ft_cfg.dp_rate, ft_cfg.weight_decay, ft_cfg.lr,
                                                                                ft_cfg.w_size, ft_cfg.stride, ft_cfg.dilate,
                                                                                ft_cfg.view, ft_cfg.data_part)
    print(ft_cfg.load_pretrained, setting)
    log_path = os.path.join(log_pth,
                            ckpt_cfg.benchmark,
                            setting,
                            model_name
    )


    writer = SummaryWriter(log_path)

    #....folder to save ckpt
    model_path = os.path.join(log_path, "model")
    os.mkdir(model_path)

    #save training config
    assert not os.path.exists(os.path.join(log_path, 'training_cfg.yaml'))
    copyfile(ckpt_cfg_pth, os.path.join(log_path, 'ckpt_cfg.yaml'))
    copyfile(args.ft_cfg_pth, os.path.join(log_path, 'ft_cfg.yaml'))

    #***********training#***********
    for epoch in range(ft_cfg.epochs):

        print("\ntraining.............")
        scheduler.step()
        train()

        torch.save(model.state_dict(),
                   '{}/epoch-{}.pth'.format(model_path, epoch + 1))