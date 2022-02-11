import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    r'''
        FocalLoss class. This loss isn't contain in Pytorch(1.10).
        this code was inherited from `https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/3`.
    
        Args
            alpha (`int or float`, *optional*, defaults to 1):
                pass

            gamma (`int or float`, *optional*, defaults to 1):
                pass

            reduce (`bool`, *optional*, defaults to True):
                Decise to mean FocalLoss output.

            class_weights (`floatTensor`, *optional*, defaults to None):
                If task dataset is imbalanced, applying class_weight to loss can way to improve model performance.(not proved!)
    '''
    def __init__(self, alpha=1, gamma=2, reduce=True, cls_weights = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

        if cls_weights is None:
            self.base_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        else:
            self.base_cross_entropy = nn.CrossEntropyLoss(weight = cls_weights, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.base_cross_entropy(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss