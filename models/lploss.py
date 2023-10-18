import torch

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, y_pred, y):

        y_pred = y_pred.flatten(start_dim=-self.d)
        y = y.flatten(start_dim=-self.d)

        diff_norms = torch.norm(y_pred - y, self.p, -1)
        y_norms = torch.norm(y, self.p, -1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, y_pred, y):
        return self.rel(y_pred, y)