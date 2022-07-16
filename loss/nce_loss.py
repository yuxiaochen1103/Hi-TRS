import torch
import torch.nn as nn

class NCELoss(nn.Module):
    def __init__(self, T=0.07):
        super(NCELoss, self).__init__()

        self.T = T
        self.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()



    def forward(self, emb_pred, emb_gt):
        # next clip matching by using nce loss

        #print(emb_pred.shape, emb_gt.shape)
        #1/0
        assert emb_pred.shape == emb_gt.shape
        assert len(emb_pred.shape) == 2

        emb_pred = nn.functional.normalize(emb_pred, dim=1)  # l2 normalize #[batch，dim]
        emb_gt = nn.functional.normalize(emb_gt, dim=1)

        sim_matrix = torch.matmul(emb_pred, emb_gt.transpose(0, 1))

        sim_matrix /= self.T
        #print('sim_matrix.shape:', sim_matrix.shape)

        labels = torch.arange(0, emb_pred.shape[0], dtype=torch.long).cuda()

        loss = self.CrossEntropyLoss(sim_matrix, labels)

        return loss, sim_matrix, labels
