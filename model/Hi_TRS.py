import sys
sys.path.append('./')
from model.TRS_submodel_pre_LN import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from model.PE_learnable import PE
S_PE=PE
T_PE=PE
C_PE=PE
from loss.mask_smooth_L1 import MaskedL2


class Hi_TRS(nn.Module):

    # d_model, dim_feedforward, nhead, d_att, dp_rate, layer_num
    def __init__(self, w_size, d_model, nhead, d_att, dp_rate, layer_num):
        super(Hi_TRS, self).__init__()


        self.j_num = 25
        self.j_dim = 3

        self.s_d_model = d_model
        self.c_d_model = d_model * 2
        self.v_d_model = d_model * 4

        print('self.s_d_model:', self.s_d_model, 'self.c_d_model:', self.c_d_model,
              'self.v_d_model:', self.v_d_model)

        self.w_size = w_size
        self.max_clip_num = 200
        self.s_nhead = nhead
        self.c_nhead = nhead
        self.v_nhead = nhead

        print('self.s_nhead:', self.s_nhead, 'self.c_nhead:', self.c_nhead,
              'self.v_nhead:', self.v_nhead)

        self.iter_num = 0

        # ********************
        # ****frame level
        # ********************
        self.s_input_map = nn.Sequential(
            nn.Linear(self.j_dim, self.s_d_model),
            nn.GELU(),
            nn.LayerNorm(self.s_d_model),
        )

        base_layer = TransformerEncoderLayer(d_model=self.s_d_model,
                                             d_inner=self.s_d_model,
                                             n_head=self.s_nhead, d_att=d_att, dropout=dp_rate)  # consisted of layernorm
        # Encoder
        self.s_enc = TransformerEncoder(base_layer, layer_num)

        # PE
        self.s_pe = S_PE((self.s_d_model, self.j_num))
        # pose regressor
        self.s_pose_reg = nn.Linear(self.s_d_model, self.j_dim)


        # ********************
        # ****clip level
        # ********************

        self.c_input_map = nn.Linear(self.s_d_model, self.c_d_model)

        base_layer = TransformerEncoderLayer(d_model=self.c_d_model,
                                             d_inner=self.c_d_model // 2,
                                             n_head=self.c_nhead, d_att=d_att, dropout=dp_rate)  # consisted of layernorm

        self.c_enc = TransformerEncoder(base_layer, layer_num)

        self.c_mask = self.get_c_mask(self.w_size, self.j_num)
        print('c_mask shape:', self.c_mask.shape)
        # PE
        self.c_pe = C_PE((self.c_d_model, self.w_size, self.j_num))
        print('c_pe shape:', self.c_pe.pe.shape)
        # infomation fusion token
        self.clip_fuse_token = nn.Parameter(torch.rand((1, 1, self.c_d_model)))
        self.c_motion_cls = nn.Linear(self.c_d_model, 2)


        # ********************
        # ****video level
        # ********************
        # Encoder
        self.v_input_map = nn.Linear(self.c_d_model, self.v_d_model)
        base_layer = TransformerEncoderLayer(d_model=self.v_d_model,
                                             d_inner=self.v_d_model // 2,
                                             n_head=self.v_nhead, d_att=d_att, dropout=dp_rate)  # consisted of layernorm

        self.v_enc = TransformerEncoder(base_layer, layer_num)
        # PE
        self.v_pe = T_PE((self.v_d_model, self.max_clip_num))
        self.video_fuse_token = nn.Parameter(torch.rand((1, 1, self.v_d_model)))

        self.v_motion_cls = nn.Linear(self.v_d_model, 2)

        #self.v_ft_reg = nn.Linear(self.v_d_model, self.w_size*self.j_num*self.j_dim)
        self.v_ft_reg = nn.Linear(self.v_d_model, self.c_d_model)

        # define loss function for each task
        self.pose_criterion = MaskedL2().cuda()

        self.c_motion_criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.v_motion_criterion = torch.nn.CrossEntropyLoss().cuda()

        print('done......')


    def get_c_mask(self, w_size, joint_num):

        mask_size = w_size * joint_num + 1  # +1 for information fusion tokens
        mask = torch.zeros(mask_size, mask_size)

        # set spatial connect as 0
        for i in range(w_size):
            begin = i * joint_num
            end = begin + joint_num

            mask[begin:end, begin:end] = 1

        # enable self-connection
        I = torch.eye(mask_size)
        mask -= I

        return mask

    def s_forward(self, input_seq):

        # output shape:
        # [batch_size, w_num, w_size, joint_num, ft_dim]

        if len(input_seq.shape) == 4: #pretrain
            batch_size, frame_num, joint_num, joint_dim = input_seq.shape
        elif len(input_seq.shape) == 5: #down stream
            batch_size, w_num, w_size, joint_num, joint_dim = input_seq.shape


        assert joint_dim == self.j_dim and joint_num == self.j_num


        x = input_seq.reshape(-1, joint_num, self.j_dim)

        # map to d_model dim
        x = self.s_input_map(x)
        x = self.s_pe(x)
        x = self.s_enc(x)  # [joint_num, -1, dim]

        if len(input_seq.shape) == 4:
            x = x.reshape(batch_size, frame_num, joint_num, self.s_d_model)

        elif len(input_seq.shape) == 5:
            x = x.reshape(batch_size, w_num, w_size, joint_num, self.s_d_model)

        return x

    def c_forward(self, x):

        # input shape:
        # batch_size, w_num, w_size, joint_num, ft_dim

        # output:    x
        # batch_size, cat_w_num, ft_dim
        x = self.c_input_map(x) #increase the input dim

        batch_size, w_num, w_size, joint_num, ft_dim = x.shape
        assert ft_dim == self.c_d_model

        x = x.reshape(-1, w_size, joint_num, ft_dim)

        x = self.c_pe(x)
        x = x.reshape(-1, w_size * joint_num, ft_dim)

        # concatenate clip_level_information fusion token
        clip_fuse_token = self.clip_fuse_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, clip_fuse_token), 1)  # [-1, w_size*joint_num+1, ft_dim]

        x = self.c_enc(x, mask=self.c_mask.cuda())

        # get embedding for each clip [batch_size*w_num, ft_dim]
        x = x.reshape(batch_size, w_num, w_size * joint_num + 1, ft_dim)
        x = x[:, :, -1, :]

        return x


    def v_forward(self, x, pad_mask):
        # input:
        # c_x: [batch_size, w_num, w_size, ft_dim]

        x = self.v_input_map(x) #increase the input dim

        batch_size, max_w_num = x.shape[:-1]

        assert pad_mask.shape == (batch_size, max_w_num + 1)
        pad_mask = pad_mask.unsqueeze(-2).unsqueeze(1)

        # add video PE
        # x = self.v_input_map(x)
        x = self.v_pe(x, length=max_w_num)

        # concatenate video_level_information fusion token
        video_fuse_token = self.video_fuse_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, video_fuse_token), 1)  # [batch_size, w_num+1, dim]

        # feed to v_enc
        assert x.shape[1] == pad_mask.shape[-1]
        x = self.v_enc(x, mask=pad_mask)

        v_cx = x[:, :-1, :]
        vx = x[:, -1, :]

        return vx, v_cx



    def cal_loss(self, s_pose_pred, org_seq, mask_seq_mask,
                 c_motion_prd, c_motion_valid_flg,
                 v_motion_prd,
                 last_cx_prd, last_cx_gt
                 #v_order_prd, clip_permute_idx, w_num_list
                 ):
        # *******************
        # ***spatial level***
        # ******************
        s_pose_loss = self.pose_criterion(pred=s_pose_pred * 10, mask_label=org_seq * 10, mask=mask_seq_mask)

        # *******************
        # ***clip level***
        # ******************
        # motion validation
        bz = s_pose_pred.shape[0]

        #cal_loss:
        assert c_motion_prd.shape[0] == 2 * bz
        c_motion_gt = torch.zeros(2 * bz).cuda().long()
        c_motion_gt[:bz] = 1
        c_motion_loss = self.c_motion_criterion(c_motion_prd, c_motion_gt)

        # add loss mask
        c_motion_loss_mask = torch.cat((c_motion_valid_flg, c_motion_valid_flg), 0)
        assert c_motion_loss_mask.shape[0] == 2 * bz
        assert c_motion_loss.shape == c_motion_loss_mask.shape
        c_motion_loss = (c_motion_loss * c_motion_loss_mask).sum() / c_motion_loss_mask.sum()


        # *******************
        # ***video level***
        # ******************
        # motion validation
        v_motion_gt = torch.zeros(2 * bz).cuda().long()
        v_motion_gt[:bz] = 1
        #1/0
        v_motion_loss = self.v_motion_criterion(v_motion_prd, v_motion_gt)

        return s_pose_loss, \
               c_motion_prd, c_motion_gt, c_motion_loss, c_motion_loss_mask, \
               v_motion_prd, v_motion_gt, v_motion_loss, \
               last_cx_prd, last_cx_gt


    def forward(self, masked_seq, org_seq, mask_seq_mask,
                    c_motion_clip, c_motion_valid_flg,
                    slid_window, permute_v, w_pad_masks,
                    last_c):

        c_forward = self.c_forward
        v_forward = self.v_forward

        # ********************
        # *******spatial level
        # ********************
        s_x = self.s_forward(masked_seq)
        s_pose_pred = self.s_pose_reg(s_x) #[batch_size, w_num, w_size, joint_num, joint_dim]
        assert s_x.shape[:-1] == s_pose_pred.shape[:-1]

        # ********************
        # *******clip level
        # ********************
        #--motion valid
        c_ord_sx = []
        batch_size = len(c_motion_clip)
        for idx in range(batch_size):
            ele = s_x[idx, c_motion_clip[idx]].unsqueeze(0) #[1, 2, w_size, joint_num, dim]
            c_ord_sx.append(ele)

        c_ord_sx = torch.cat(c_ord_sx, 0)
        assert c_ord_sx.shape[:-1] == (batch_size, 2, self.w_size, self.j_num)

        #--get  embeding for each clip
        c_ord_cx = c_forward(c_ord_sx)
        assert c_ord_cx.shape[:-1] == (batch_size, 2)

        #--make prediction
        c_motion_prd = self.c_motion_cls(torch.cat([c_ord_cx[:, 0], c_ord_cx[:, 1]], 0)).squeeze()

        # ********************
        # ***video level
        # ********************
        #use sliding window
        v_s_x = s_x[:, slid_window]
        assert v_s_x.shape[:-1] == (batch_size, len(slid_window), self.w_size, self.j_num)
        #extract embedding for each clip
        c_x = c_forward(v_s_x)
        assert c_x.shape[:-1] == (batch_size, len(slid_window)) and len(c_x.shape) == 3
        #get video embedding
        vx_org, v_cx_org = v_forward(c_x, w_pad_masks)
        assert vx_org.shape[0] == batch_size and len(vx_org.shape) == 2

        #----feature regression
        #get ground_truth
        last_c_sx = []
        for idx in range(batch_size):
            #s_x: bacth, w_num, w_size, joint_num,
            #s_x, batch
            ele = s_x[idx, last_c[idx]].unsqueeze(0)  # [1, w_size, joint_num, dim]
            last_c_sx.append(ele)
        last_c_sx = torch.cat(last_c_sx, 0).unsqueeze(1) # (batch_size, 1, self.w_size, self.j_num)
        assert last_c_sx.shape[:-1] == (batch_size, 1, self.w_size, self.j_num)
        last_cx_gt = c_forward(last_c_sx).squeeze().detach()
        #get_prd
        last_cx_prd = self.v_ft_reg(vx_org).squeeze()

        # permute
        permute_v_cx = []
        for b_i, c_idx in enumerate(permute_v):
            permute_v_cx.append(c_x[b_i, c_idx, :].unsqueeze(0))

        permute_v_cx = torch.cat(permute_v_cx, 0)
        assert c_x.shape == permute_v_cx.shape
        vx_permute, _ = v_forward(permute_v_cx, w_pad_masks)


        v_motion_prd = self.v_motion_cls(torch.cat([vx_org, vx_permute], 0)).squeeze()


        #print('org_shuffle_prd.shape, permute_shuffle_prd.shape:', org_shuffle_prd.shape, permute_shuffle_prd.shape)

        return self.cal_loss(s_pose_pred=s_pose_pred, org_seq=org_seq, mask_seq_mask=mask_seq_mask,
                             c_motion_prd=c_motion_prd, c_motion_valid_flg=c_motion_valid_flg,
                            v_motion_prd=v_motion_prd,
                            last_cx_gt=last_cx_gt, last_cx_prd=last_cx_prd)



    def clip_cls_forward(self, x):

        #x: bacth_size, w_num, person_num, joint_num, joint_dim
        assert len(x.shape) == 5 and x.shape[1:] == (2, self.w_size, self.j_num, self.j_dim)

        batch_size = x.shape[0]

        #merge human person dim
        #print(x.shape, batch_size)
        x = x.view(-1, self.w_size, self.j_num, self.j_dim)
        #print(x.shape)
        x = x.unsqueeze(1) #add window_size dim
        #print(x.shape, (batch_size*2, 1, self.w_size, self.j_num, self.j_dim))
        assert x.shape == (batch_size*2, 1, self.w_size, self.j_num, self.j_dim)

        # input shape:

        # concate with next filp

        x = self.s_forward(x)
        #print(x.shape)
        x = self.c_forward(x)
        #print(x.shape)
        x = x.view(batch_size, 2, x.shape[-1])
        #print(x.shape)
        x = x.sum(1) / 2
        #print(x.shape)
        assert x.shape == (batch_size, x.shape[-1])


        # feed to FC layer for classifer
        score = self.c_cls(x)

        return score.squeeze()



if __name__ == "__main__":

    # w_size, d_model, nhead, dim_feedforward, dp_rate, layer_num
    #w_size, d_model, nhead, d_att, dp_rate, layer_num
    model = SKL_TRS(w_size=5, d_model=2, nhead=1,
                    dim_feedforward=8, dp_rate=0.5, layer_num=2).cuda()

    model = model.cuda()
