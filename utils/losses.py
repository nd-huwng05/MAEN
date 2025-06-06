import torch
import torch.nn.functional as F
import torch.nn as nn

class CKA(nn.Module):
    def __init__(self, device='cpu'):
        super(CKA, self).__init__()
        self.device = device

    def center_gram(self, gram):
        """Centering the Gram matrix."""
        n = gram.size(0)
        unit = torch.ones((n, n), device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return H @ gram @ H

    def gram_linear(self, x):
        """Compute Gram matrix using linear kernel (XXᵀ)."""
        return x @ x.T

    def hsic(self, x, y):
        """Compute HSIC between two matrices."""
        K = self.center_gram(self.gram_linear(x))
        L = self.center_gram(self.gram_linear(y))
        return (K * L).sum()

    def forward(self, x, y):
        """
        Compute CKA similarity between x and y.
        x, y: tensors of shape (batch_size, features)
        """
        hsic_xy = self.hsic(x, y)
        hsic_xx = self.hsic(x, x)
        hsic_yy = self.hsic(y, y)
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)
        return cka


class MMD:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _gaussian_kernel(self, x, y):
        # x: (B, D), y: (B, D)
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)
        dist = x_norm + y_norm - 2 * torch.mm(x, y.t())
        k = torch.exp(-self.gamma * dist)
        return k

    def compute(self, x, y):
        """
        x, y: Tensors with shape (B, D)
        """
        K_xx = self._gaussian_kernel(x, x)
        K_yy = self._gaussian_kernel(y, y)
        K_xy = self._gaussian_kernel(x, y)

        B = x.size(0)
        if B > 1:
            K_xx = (K_xx.sum() - K_xx.diag().sum()) / (B * (B - 1))
            K_yy = (K_yy.sum() - K_yy.diag().sum()) / (B * (B - 1))
        else:
            K_xx = K_xx.mean()
            K_yy = K_yy.mean()

        K_xy = K_xy.mean()

        return K_xx + K_yy - 2 * K_xy


class Cosine:
    @staticmethod
    def distance(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 1 - torch.sum(x * y) / x.size(0)


class SimilarityLoss:
    def __init__(self, alpha=0.7, beta=0.3, lambda_f=0.8, lambda_a=0.2, device='cpu', idx=0):
        self.alpha = alpha
        self.beta = beta
        self.lambda_f = lambda_f
        self.lambda_a = lambda_a

        self.idx = idx
        self.cka = CKA(device=device)
        self.mmd = MMD()

    def feature_level_loss(self, features):
        n = len(features)
        loss = 0
        for j in range(self.idx + 1, n):
            h_j = features[j]
            h_n = features[self.idx]
            cka_loss = self.cka(h_j.float(), h_n.float())
            mmd_loss = 1 - self.mmd.compute(h_j, h_n)
            loss += self.alpha * cka_loss + self.beta * mmd_loss
        return loss

    def attention_level_loss(self, attentions):
        n = len(attentions)
        loss = 0
        for j in range(self.idx +1, n):
            A_j = attentions[j].mean(dim=0).flatten()
            A_n = attentions[self.idx].mean(dim=0).flatten()
            loss += Cosine.distance(A_j.float(), A_n.float())
        return loss

    def __call__(self, features, attentions):
        loss_f = self.feature_level_loss(features)
        loss_a = self.attention_level_loss(attentions)

        total_loss = self.lambda_f * loss_f + self.lambda_a * loss_a
        return total_loss

