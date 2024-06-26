import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.output_layer(x)
        return x

    def loss(self, x, y):
        y_hat = self(x)
        l2_norm = self.output_layer.weight.pow(2).sum()
        loss = self.loss_fn(y_hat, y) + self.alpha * l2_norm
        return loss


