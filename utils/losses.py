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

    def forward(self, X, Y, group_size=2):
        assert X.shape == Y.shape, "X and Y must have the same shape"

        batch_size = X.shape[0]
        num_full_groups = batch_size // group_size
        remainder = batch_size % group_size

        assert remainder != 1

        total = 0.0
        start = 0

        for i in range(num_full_groups):
            end = start + group_size
            x_chunk = X[start:end].flatten(1)
            y_chunk = Y[start:end].flatten(1)
            total += self.linear_CKA(x_chunk, y_chunk)
            start = end

        # Gộp phần còn lại (nếu có)
        if remainder > 0:
            x_chunk = X[start:].flatten(1)
            y_chunk = Y[start:].flatten(1)
            total += self.linear_CKA(x_chunk, y_chunk)

        return total



class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y, Y_rec):
        return torch.mean((Y - Y_rec)**2, dim=-1)

class AEULoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1 ,args=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.cka = CKA()
        self.recon = ReconstructionLoss()

    def sim(self, features):
        loss = 0
        for idx in range(len(features)):
            for i in range(idx + 1, len(features)):
                loss += self.cka.forward(features[idx], features[i])
        return loss

    def rec_loss(self, image, recs, log_vars):
        loss = 0
        for i in range(len(recs)):
            rec = sum(self.recon(image, recs[i]))
            loss += rec
        return torch.mean(loss)

    def forward(self, x_recons, features, image, log_vars):
        L_sim = self.sim(features)
        L_rec = self.rec_loss(image, x_recons, log_vars)
        # L_log = self.log_loss(log_vars)
        return L_sim, L_rec, L_sim*self.alpha + L_rec*self.beta

class MAENLoss(nn.Module):
    def __init__(self, alpha=1, beta=1 ,args=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cka = CKA()
        self.recon = ReconstructionLoss()

    def sim(self, features):
        loss = 0.0
        for idx in range(len(features)):
            for i in range(idx + 1, len(features)):
                loss += self.cka.forward(features[idx], features[i], group_size=8)
        return loss

    def rec_loss(self, image, recs, mask):
        loss = 0.0
        for i in range(len(recs)):
            rec = (self.recon(image, recs[i])*mask[i]).sum()/mask[i].sum()
            loss += rec
        return loss

    def rec_loss_uncertainty(self, image, x_recons, log_vars, masks):
        """
        Heteroscedastic loss: exp(-log_var) * (x - x_hat)^2 + log_var
        """
        total_loss = 0.0
        for i in range(len(x_recons)):
            x_hat = x_recons[i]
            log_var = log_vars[i]
            mask = masks[i]
            rec_err = (image - x_hat) ** 2
            hetero_loss = torch.exp(-log_var) * rec_err + log_var
            masked_loss = (hetero_loss * mask).sum() / mask.sum()
            total_loss += masked_loss
        return total_loss

    def forward(self, x_recons, features, image, log_vars, masks, uncertainty = False):
        L_sim = self.sim(features)
        if uncertainty:
            L_rec = self.rec_loss_uncertainty(image, x_recons, log_vars, masks)
        else:
            L_rec = self.rec_loss(image, x_recons, masks)
        return L_sim, L_rec, L_sim*self.alpha + L_rec*self.beta