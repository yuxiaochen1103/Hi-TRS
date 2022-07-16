import torch
import torch.nn as nn

class MaskedL2(nn.Module):
    def __init__(self):
        super(MaskedL2, self).__init__()
        self.mask_element = torch.tensor([999, 999, 999]).cuda()
        #self.SmoothL1Los = nn.SmoothL1Loss(reduction='none').cuda()
        self.SmoothL1Los = nn.SmoothL1Loss().cuda()

    def forward(self, pred, mask_label, mask):

        assert pred.shape == mask_label.shape

        pose_dim = pred.shape[-1]

        pred = pred.reshape(-1, pose_dim)
        mask_label = mask_label.reshape(-1, pose_dim)
        mask = mask.reshape(-1, 1)
        #
        assert pred.shape == mask_label.shape
        assert pred.shape[0] == mask.shape[0]

        #loss = (((pred - mask_label)**2.0).sum(-1).sqrt() * mask).sum() / torch.sum(mask)
        # loss = self.SmoothL1Los(pred, mask_label)
        # assert loss.shape[0] == mask.shape[0]
        # loss = (loss * mask).sum() / torch.sum(mask) / pred.shape[-1]
        # return loss
        #print(pred.shape,mask.sum())
        #print(pred[mask.long()].shape)
        pred = torch.masked_select(pred, mask==1).view(-1,3)
        mask_label = torch.masked_select(mask_label, mask == 1).view(-1,3)
        #print(pred.shape)

        assert pred.shape[0] == mask.sum() and pred.shape[1] == 3 and len(pred.shape) == 2
        assert mask_label.shape[0] == mask.sum() and mask_label.shape[1] == 3 and len(mask_label.shape) == 2

        loss = self.SmoothL1Los(pred, mask_label)

        return loss


