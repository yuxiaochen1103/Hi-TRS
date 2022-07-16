import torch
from dataset.NTU_3D_pretrain import NTU_Pretrain, my_collate
import yaml
import torch.optim as optim
from model.Hi_TRS import Hi_TRS
import time
import argparse
from tensorboardX import SummaryWriter
import os
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import random
from Optim import get_polynomial_decay_schedule_with_warmup
from loss.nce_loss import NCELoss

parser = argparse.ArgumentParser()

parser.add_argument("--cfg_pth")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_data_loader(cfg):

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    # benchmark, data_split, w_size, stride, use_data_aug)
    train_dataset = NTU_Pretrain(w_size=cfg.w_size, stride=cfg.stride, dilate=cfg.dilate,
                                 benchmark=cfg.benchmark, view=cfg.view, label_num=cfg.label_num,
                                 use_data_aug=True, data_split='train')

    val_dataset = NTU_Pretrain(w_size=cfg.w_size, stride=cfg.stride, dilate=cfg.dilate,
                               benchmark=cfg.benchmark, view=cfg.view, label_num=cfg.label_num,
                               use_data_aug=False, data_split='val')


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        collate_fn=my_collate,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=cfg.workers,
        collate_fn=my_collate,
        pin_memory=True,
        shuffle=False, )

    return train_loader, val_loader


def init_model(cfg):
    # joint_num, w_size, d_model, nhead, dim_feedforward, dp_rate, layer_num
    model = Hi_TRS(w_size=cfg.w_size,
                    d_model=cfg.d_model,
                    nhead=cfg.nhead, d_att=cfg.d_att,
                    layer_num=cfg.layer_num,
                    dp_rate=cfg.dp_rate
                   )

    model = torch.nn.DataParallel(model).cuda().float()

    return model


def get_acc(score, labels, mask=None):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    #print(outputs)
    #print(labels)
    #print(outputs == labels)

    # print(score.shape, labels.shape, outputs.shape)
    if mask is not None:
        mask = mask.cpu().data.numpy()
        outputs = outputs[mask == 1]
        labels = labels[mask == 1]
        assert outputs.shape[0] == mask.sum()

    assert outputs.shape == labels.shape

    #print(np.sum(outputs == labels))

    #1/0

    return np.sum(outputs == labels) / float(labels.size)


def forward(model, batch):
    # 'masked_seq': masked_seq, "org_seq": org_seq, 'mask_seq_mask': mask_seq_mask,
    # 'c_motion_clip': c_motion_clip, 'c_motion_valid_flg': c_motion_valid_flg,
    # 'permute_v': permute_v,
    # 'slid_window': slid_window, 'w_pad_mask': w_pad_mask, 'wnum_list': wnum_list,
    # 'last_c': last_c

    #frame_level
    masked_seq = batch['masked_seq'].cuda().float()
    org_seq = batch["org_seq"].cuda().float()
    mask_seq_mask = batch['mask_seq_mask'].cuda()

    #-------clip level
    c_motion_valid_flg = batch['c_motion_valid_flg'].long().cuda()
    c_motion_clip = batch['c_motion_clip']

    # -------video level
    # clip
    slid_window = batch['slid_window']
    permute_v = batch['permute_v'].cuda().long()

    # length info
    w_pad_mask = batch['w_pad_mask'].bool().cuda()

    last_c = batch['last_c'].cuda()

    s_pose_loss, \
    c_motion_prd, c_motion_gt, c_motion_loss, c_motion_loss_mask, \
    v_motion_prd, v_motion_gt, v_motion_loss, \
    last_cx_prd, last_cx_gt = model(masked_seq=masked_seq, org_seq=org_seq, mask_seq_mask=mask_seq_mask,
                c_motion_clip=c_motion_clip, c_motion_valid_flg=c_motion_valid_flg,
                slid_window=slid_window, permute_v=permute_v, w_pad_masks=w_pad_mask,
                last_c=last_c
                )
    if len(s_pose_loss) > 0:
        batch_size = org_seq.shape[0]
        #spose
        s_pose_loss = s_pose_loss.mean()

        #frame order
        c_motion_loss = c_motion_loss.mean()
        assert c_motion_prd.shape[0] == c_motion_gt.shape[0] == c_motion_loss_mask.shape[0] == batch_size * 2

        #for video level
        #clip_shuffle
        v_motion_loss = v_motion_loss.mean()
        assert v_motion_prd.shape[0] == v_motion_gt.shape[0] == 2 * batch_size

    #increase negative sample size
    last_cx_loss, last_cx_prd, last_cx_gt = last_clip_criterion(emb_pred=last_cx_prd,
                                                                               emb_gt=last_cx_gt)
    assert last_cx_prd.shape[0] == last_cx_gt.shape[0] == batch_size


    return s_pose_loss, \
           c_motion_prd, c_motion_gt, c_motion_loss, c_motion_loss_mask, \
           v_motion_prd, v_motion_gt, v_motion_loss, \
           last_cx_prd, last_cx_gt, last_cx_loss



loss_name_list = ['s_pose_loss',
                  'c_motion_loss', 'c_motion_acc',
                  'v_motion_loss', 'v_motion_acc',
                  'last_cx_loss', 'last_cx_acc',
                  'loss']

def train(gloal_iter):
    model.train()

    # use to record training status
    lossname_2_id = {}
    for i, name in enumerate(loss_name_list):
        lossname_2_id[name] = i

    start_time = time.time()
    temp_train_log = 0.  # every 10 iteration

    optimizer.zero_grad()  # zero grad

    i = 0
    for batch in train_loader:

        #skip data where there is no valid clip
        c_motion_valid_flg = batch['c_motion_valid_flg']
        gpu_num = torch.cuda.device_count()
        each_gpu_data_num = len(c_motion_valid_flg) // gpu_num
        is_valid = True
        for gpu_id in range(gpu_num):
            b_idx = gpu_id * each_gpu_data_num
            e_idx = (gpu_id + 1) * each_gpu_data_num
            #print(gpu_num, b_idx, e_idx, c_motion_valid_flg[b_idx:e_idx].sum())
            assert c_motion_valid_flg[b_idx:e_idx].sum() <= each_gpu_data_num
            if c_motion_valid_flg[b_idx:e_idx].sum() == 0:
                is_valid = False
                break
        if not is_valid:
            continue

        s_pose_loss, \
        c_motion_prd, c_motion_gt, c_motion_loss, c_motion_loss_mask, \
        v_motion_prd, v_motion_gt, v_motion_loss, \
        last_cx_prd, last_cx_gt, last_cx_loss = forward(model, batch)

        assert not torch.isnan(c_motion_loss)
        #and not torch.isnan(clip_cnt_loss)

        # acc
        # clip-level---
        c_motion_acc = get_acc(score=c_motion_prd, labels=c_motion_gt, mask=c_motion_loss_mask)
        #print(c_motion_gt)

        # # video-level
        # # shuflle acc
        v_motion_acc = get_acc(score=v_motion_prd, labels=v_motion_gt)

        #last
        last_cx_acc = get_acc(score=last_cx_prd, labels=last_cx_gt)

        loss = None
        if cfg.s_pose_w != 0:
            if loss is not None:
                loss = loss + cfg.s_pose_w * s_pose_loss
            else:
                loss = cfg.s_pose_w * s_pose_loss

        #
        if cfg.c_motion_w != 0:
            if loss is not None:
                loss = loss + cfg.c_motion_w * c_motion_loss
            else:
                loss = cfg.c_motion_w * c_motion_loss


        if cfg.v_motion_w != 0:
            if loss is not None:
                loss = loss + cfg.v_motion_w * v_motion_loss
            else:
                loss = cfg.v_motion_w * v_motion_loss

        if cfg.last_cx_w != 0:
            if loss is not None:
                loss = loss + cfg.last_cx_w * last_cx_loss
            else:
                loss = cfg.last_cx_w * last_cx_loss

        temp_train_log += np.array([s_pose_loss.item(),
                                    c_motion_loss.item(), c_motion_acc,
                                    v_motion_loss.item(), v_motion_acc,
                                    last_cx_loss.item(), last_cx_acc,
                                    loss.item()])

        # backward

        loss /= cfg.accum_iter
        loss.backward()

        i += 1
        #print(i)

        if i % cfg.accum_iter == 0:

            # count training iteration number
            gloal_iter += 1

            # update model paramters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            # show training log
            train_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            temp_train_log /= cfg.accum_iter
            print("*** Global_step: {} lr: {} train_time: {} ".format(gloal_iter,
                                                                      round(current_lr, 8),
                                                                      round(train_time, 3), ), end='')
            for loss_name in loss_name_list:
                loss_id = lossname_2_id[loss_name]
                print("{} {:.4f} ".format(loss_name, temp_train_log[loss_id]), end='')
                writer.add_scalar('Train/{}'.format(loss_name), temp_train_log[loss_id], gloal_iter)
            print()


            # update learning rate
            optim_scheduler.step()

            # validation
            if gloal_iter % cfg.eval_step == 0:
                with torch.no_grad():
                    epoch_val_log = evaluate(gloal_iter)

                s_pose_loss, \
                c_motion_loss, c_motion_acc, \
                v_motion_loss, v_motion_acc, \
                last_cx_loss, last_cx_acc, \
                val_loss = epoch_val_log


                torch.save(model.state_dict(),
                           '{}/iter-{}_'
                           'train-loss-{:.4f}_'
                           'val-loss-{:.4f}_'
                           's-pose-{:.4f}_'
                           'c-motion-acc-{:.4f}_'
                           'v-motion-acc-{:.4f}_'
                           'last_cx_acc-{:.4f}.pth'
                           .format(model_path, gloal_iter, temp_train_log[-1], val_loss,
                                s_pose_loss, c_motion_acc, v_motion_acc, last_cx_acc))


                print('saving model......')

            # reset training log
            temp_train_log = 0
            start_time = time.time()

            if i + cfg.accum_iter >= len(train_loader):
                break

    return gloal_iter


def evaluate(gloal_iter):
    model.eval()
    print('doing evaluataion....')


    lossname_2_id = {}
    for i, name in enumerate(loss_name_list):
        lossname_2_id[name] = i

    epoch_train_log = 0.  # whole epoch

    for i, batch in enumerate(tqdm(val_loader, total=len(val_loader))):
        s_pose_loss, \
        c_motion_prd, c_motion_gt, c_motion_loss, c_motion_loss_mask, \
        v_motion_prd, v_motion_gt, v_motion_loss, \
        last_cx_prd, last_cx_gt, last_cx_loss = forward(model, batch)

        # acc
        # clip-level
        c_motion_acc = get_acc(score=c_motion_prd, labels=c_motion_gt, mask=c_motion_loss_mask)

        # video-level
        # shuflle acc
        v_motion_acc = get_acc(score=v_motion_prd, labels=v_motion_gt)
        last_cx_acc = get_acc(score=last_cx_prd, labels=last_cx_gt)

        loss = 0
        if cfg.s_pose_w != 0:
            loss += cfg.s_pose_w * s_pose_loss

        if cfg.c_motion_w != 0:
            loss += cfg.c_motion_w * c_motion_loss

        if cfg.v_motion_w != 0:
            loss += cfg.v_motion_w * v_motion_loss

        if cfg.last_cx_w != 0:
            loss = loss + cfg.last_cx_w * last_cx_loss



        epoch_train_log += np.array([s_pose_loss.item(),
                                    c_motion_loss.item(), c_motion_acc,
                                    v_motion_loss.item(), v_motion_acc,
                                    last_cx_loss.item(), last_cx_acc,
                                    loss.item()])
    # save result to tensorboard every epoch

    epoch_train_log /= (i + 1)
    for loss_name in loss_name_list:
        loss_id = lossname_2_id[loss_name]
        writer.add_scalar('Val/{}'.format(loss_name), epoch_train_log[loss_id], gloal_iter)
    return epoch_train_log


seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":

    # *** load config
    args = parser.parse_args()

    cfg_pth = args.cfg_pth
    cfg_file = open(cfg_pth)
    cfg = yaml.load(cfg_file)
    cfg = AttrDict(cfg)

    print('-' * 30)
    print(args)
    print(cfg)
    print('-' * 30)
    print()


    # .... get dataloader
    train_loader, val_loader = get_data_loader(cfg)

    # .........inital model
    print("\ninit model.............")
    model = init_model(cfg)

    #no_wd_list = []
    #wd_list = []
    no_decay_name_list = ["bias", "LayerNorm", '_norm', 's_input_map.2.weight', '_pe.', 'fuse_token']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_name_list) and p.requires_grad],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_name_list) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.lr)

    optim_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=cfg.warmup_steps,
                                                                num_training_steps=cfg.train_step, lr_end=cfg.lr_end
                                                                )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=cfg.lr_gamma, step_size=150, verbose=True)
    last_clip_criterion = NCELoss().cuda()
    # pose_criterion = MaskedSmoothL1().cuda()
    # ....tensorboard logger
    log_path = os.path.join(cfg.log_pth,
                            cfg.benchmark,
                            'dim-{}_nhead-{}_d-att-{}_layer-{}_'
                            'w-size-{}_strde-{}_dilate-{}_'
                            'dp-rate-{}_wd-{}_lr-{}'.format(
                                                            cfg.d_model, cfg.nhead, cfg.d_att, cfg.layer_num,
                                                            cfg.w_size, cfg.stride, cfg.dilate,
                                                            cfg.dp_rate, cfg.weight_decay, cfg.lr),
                            's-pose-{}_'
                            'c-motion-{}_'
                            'v-motion-{}_last-cx-{}_view-{}'.format(
                                cfg.s_pose_w,
                                cfg.c_motion_w,
                                cfg.v_motion_w, cfg.last_cx_w,
                                cfg.view
                            ))
    print(log_path)
    writer = SummaryWriter(log_path)

    # ....folder to save ckpt
    model_path = os.path.join(log_path, "model")
    os.mkdir(model_path)

    # save training config
    assert not os.path.exists(os.path.join(log_path, 'training_cfg.yaml'))
    copyfile(cfg_pth, os.path.join(log_path, 'training_cfg.yaml'))

    # parameters recording training log
    no_improve_epoch = 0
    gloal_iter = 0
    min_loss = 99999

    # ***********training#***********
    while 1:
        # print("\ntraining.............")
        gloal_iter = train(gloal_iter)  # [s_pose, t_pose, loss]

        if gloal_iter > cfg.train_step:
            break
