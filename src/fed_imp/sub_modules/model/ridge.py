import torch
import torch.nn as nn


class RidgeRegression(nn.Module):
	def __init__(self, input_dim, alpha):
		super(RidgeRegression, self).__init__()
		self.output_layer = nn.Linear(input_dim, 1)
		self.alpha = alpha
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		x = self.output_layer(x)
		return x

	def loss(self, x, y):
		y_hat = self(x)
		l2_norm = self.output_layer.weight.pow(2).sum()
		loss = self.loss_fn(y_hat, y) + self.alpha * l2_norm
		return loss


def ridge_loss(Y, pred, w, lamb):
	pred_loss = torch.norm((Y - pred), p='fro') ** 2
	reg = torch.norm(w, p='fro') ** 2
	return (1 / Y.size()[0]) * pred_loss + lamb * reg


def fit(lamb, X_pt, Y_pt, w, epochs=3, learning_rate=0.1):
	w_pt = torch.tensor(w, requires_grad=True)
	opt = torch.optim.Adam([w_pt], lr=learning_rate, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(epochs):
		pred = torch.matmul(X_pt, w_pt)
		loss = ridge_loss(Y_pt, pred, w_pt, lamb)
		loss.backward()
		opt.step()
		opt.zero_grad()
	return w_pt


# X_pt = torch.from_numpy(X)  # xtrain
# Y_pt = torch.from_numpy(Y)  # ytrain
# Y_ptt = torch.from_numpy(Y_test)  # xtest
# X_ptt = torch.from_numpy(X_test)  # ytest
# w = np.random.rand(X.shape[1], Y.shape[1])
# weight = fit(0.001, X_pt, Y_pt, w)
