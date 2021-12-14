import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_gen, features_real):
        loss = 0

        for f_gen, f_real in zip(features_gen, features_real):
            for fg, fr in zip(f_gen, f_real):
                loss += F.l1_loss(fg, fr)

        return loss


class AdversarialLoss(nn.Module):
    def __init__(self, real_label=1, fake_label=0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label

    def calc_discriminator_loss(self, gen_logits, real_logits):
        loss = 0
        for gl, rl in zip(gen_logits, real_logits):
            mse_real = ((gl - self.real_label) ** 2).mean()
            mse_gen = ((rl - self.fake_label) ** 2).mean()
            loss += (mse_real + mse_gen)

        return loss * 0.5

    def calc_generator_loss(self, gen_logits):
        loss = 0

        for logits in gen_logits:
            loss += ((logits - self.real_label)**2).mean()

        return loss
