import torch
import math
import torch.nn as nn
from torch.ao.nn.quantized.functional import clamp


class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        eps=1e-8
        return hsic / max((var1 * var2),eps)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

class MAENLoss(nn.Module):
    ema_loss = 0
    def __init__(self, lamb_max=2):
        super().__init__()
        self.cka = CudaCKA(device="cuda")
        self.lamb_max = lamb_max


    def EMA(self, loss, alpha=0.9):
        return alpha*MAENLoss.ema_loss + (1.0-alpha)*loss

    def lamb(self, rec_err, div_err, e=1e-6, r=1):
        return (self.EMA(rec_err)/(self.EMA(div_err) + e))*r

    def CKA(self, x, y):
        return self.cka.linear_CKA(x.flatten(1), y.flatten(1))

    def rec_loss(self, img, pred, mask):
        rec_err = (img - pred)**2
        rec_err = rec_err.mask.sum()/mask().sum()
        return rec_err

    def div_loss(self, cls_tokens):
        total = 0.0
        for idx in range(len(cls_tokens)):
            for i in range(idx+1, len(cls_tokens)):
                total += self.CKA(cls_tokens[idx], cls_tokens[i])
        return total

    def forward(self, imgs, preds, cls_tokens, masks):
        L_rec = 0.0
        L_div = self.div_loss(cls_tokens)
        for i in range(len(preds)):
            L_rec += self.rec_loss(imgs[i], preds[i], masks[i])
        lamb = clamp(self.lamb, 0 , self.lamb_max)
        return L_rec + lamb*L_div