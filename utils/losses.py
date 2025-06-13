import torch
import math
import torch.nn as nn

class CKA(object):
    def __init__(self, device='cuda'):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device, dtype=K.dtype)
        I = torch.eye(n, device=self.device, dtype=K.dtype)
        H = I - unit / n
        return H @ K @ H

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX).unsqueeze(0) - GX
        KX = KX + KX.T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = mdist.sqrt()
        KX = -0.5 * KX / (sigma * sigma)
        return torch.exp(KX)

    def kernel_HSIC(self, X, Y, sigma):
        KX = self.centering(self.rbf(X, sigma))
        KY = self.centering(self.rbf(Y, sigma))
        return torch.sum(KX * KY)

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        eps = 1e-8
        return hsic / max(var1 * var2, eps)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

    def forward(self, X, Y):
        return self.linear_CKA(X.flatten(1), Y.flatten(1))


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y, Y_rec):
        return torch.mean((Y - Y_rec) ** 2)


class MAENLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1 ,args=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.cka = CKA()
        self.recon = ReconstructionLoss()

    def sim(self, features):
        loss = []
        for idx in range(len(features)):
            for i in range(idx + 1, len(features)):
                loss.append(self.cka.forward(features[idx], features[i]))
        return torch.mean(torch.stack(loss), dim=0)

    def rec_loss(self, image, recs, log_vars):
        loss = 0
        for i in range(len(recs)):
            rec = self.recon(image, recs[i])
            loss += torch.mean(torch.exp(-log_vars[i] * rec))
        return loss

    def log_loss(self, log_vars):
        loss = 0
        for i in range(len(log_vars)):
            loss += torch.mean(log_vars[i])
        return loss

    def forward(self, x_recons, features, image, log_vars):
        L_sim = self.sim(features)
        L_rec = self.rec_loss(image, x_recons, log_vars)
        L_log = self.log_loss(log_vars)
        return L_sim, L_rec, L_sim*self.alpha + L_rec*self.beta + L_log*self.gamma