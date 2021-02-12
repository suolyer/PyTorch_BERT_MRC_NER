import torch
from torch import nn
device = torch.device('cuda')


class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels,seq_mask):
        logits=logits[:,:,1]
        ones=torch.ones_like(logits)
        zero=torch.zeros_like(logits)
        y_pred=torch.where(logits<0.5,zero,ones)
        y_pred=y_pred.view(size=(-1,)).float()

        seq_mask=seq_mask.view(size=(-1,)).float()
        # y_pred=torch.multiply(y_pred,seq_mask)

        y_true=labels.view(size=(-1,)).float()
        corr=torch.eq(y_pred,y_true)
        corr=torch.multiply(corr.float(),y_true)
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        f1=2*recall*precision/(recall+precision+1e-8)
        return recall, precision, f1


class metrics_start(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels,seq_id):
        logits=logits[:,:,1]
        batch_size,max_length=logits.shape
        y_pred=torch.Tensor(1,max_length).to(device)
        for i in range(batch_size):

            tmp=logits[i]
            SEP=tmp[seq_id[i][0]]
            tmp=tmp.view(1,-1)
            
            ones=torch.ones_like(tmp)
            zero=torch.zeros_like(tmp)
            tmp_y_pred=torch.where(tmp<=SEP,zero,ones)
            y_pred=torch.cat([y_pred,tmp_y_pred],axis=0)

        y_pred=y_pred[1:,:]
        y_pred=y_pred.view(size=(-1,)).float()
        y_true=labels.contiguous().view(size=(-1,)).float()
        corr=torch.eq(y_pred,y_true)
        corr=torch.multiply(corr.float(),y_true)
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        f1 = 2*recall*precision/(recall+precision+1e-8) 
        return recall, precision, f1


class metrics_end(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels,seq_id):
        logits=logits[:,:,1]
        batch_size,max_length=logits.shape
        y_pred=torch.Tensor(1,max_length).to(device)
        for i in range(batch_size):
            tmp=logits[i]
            SEP=tmp[seq_id[i][1]]
            tmp=tmp.view(1,-1)
            ones=torch.ones_like(tmp)
            zero=torch.zeros_like(tmp)
            tmp_y_pred=torch.where(tmp<=SEP,zero,ones)
            y_pred=torch.cat([y_pred,tmp_y_pred],axis=0)

        y_pred=y_pred[1:,:]
        y_pred=y_pred.view(size=(-1,)).float()
        y_true=labels.contiguous().view(size=(-1,)).float()
        corr=torch.eq(y_pred,y_true)
        corr=torch.multiply(corr.float(),y_true)
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        f1 = 2*recall*precision/(recall+precision+1e-8)

        return recall, precision, f1