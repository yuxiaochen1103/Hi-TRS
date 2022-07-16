import sys
sys.path.append('./')
from model.Hi_TRS import Hi_TRS
import torch


class Hi_TRS_AR(Hi_TRS):

    # d_model, dim_feedforward, nhead, d_att, dp_rate, layer_num
    def __init__(self, w_size, d_model, nhead, d_att, dp_rate, layer_num, label_num):
        super().__init__(w_size, d_model, nhead, d_att, dp_rate, layer_num)

        self.v_cls = torch.nn.Linear(d_model * 4, label_num)

    def forward(self, x, w_pad_mask):
        #c_forward = self.c_forward_no_token
        #v_forward = self.v_forward_no_token

        c_forward = self.c_forward
        v_forward = self.v_forward

        # input shape:
        #x: [batch_size, window_num, self.w_size, 2, self.joint_num, self.j_dim]

        batch_size, w_num, w_size, body_num, j_num, j_dim = x.shape
        #print(x.shape)
        assert w_size == self.w_size and j_num == self.j_num and j_dim == self.j_dim and body_num == 2

        x = x.permute(0,3,1,2,4,5).contiguous() #--> [batch_size, 2, window_num, self.w_size, self.joint_num, 3]
        assert x.shape == (batch_size, body_num, w_num, w_size, j_num, j_dim)

        x = x.view(batch_size * body_num, w_num, w_size, j_num, j_dim)
        w_pad_mask = w_pad_mask.view(batch_size * body_num, w_num+1)
        x = self.s_forward(x)
        x = c_forward(x)
        vx, v_cx = v_forward(x, w_pad_mask)

        x = vx
        x = x.view(batch_size, body_num, self.v_d_model)

        x = x.sum(1) / 2

        # feed to FC layer for classifer
        score = self.v_cls(x)

        return score

if __name__ == "__main__":
    #w_size, d_model, nhead, d_att, dp_rate, layer_num, label_num
    model = Hi_TRS_AR(w_size=5, d_model=2, nhead=1, d_att=16,
                    dp_rate=0.5, layer_num=2, label_num=60).cuda()

    model = model.cuda()