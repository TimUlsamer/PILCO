import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiOutputGP:
    def __init__(self, input_dim, output_dim):
        self.models = []
        self.likelihoods = []
        for _ in range(output_dim):
            # Dummy Data zum Starten, wird beim Training überschrieben
            x_init = torch.zeros(2, input_dim)
            y_init = torch.zeros(2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(x_init, y_init, likelihood)
            self.models.append(model)
            self.likelihoods.append(likelihood)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def train(self, X, Y, training_iter=50):
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            model.set_train_data(X, Y[:, i], strict=False)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
            ], lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(training_iter):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, Y[:, i])
                loss.backward()
                optimizer.step()

    def predict(self, Xtest):
        preds = []
        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()
            # Kein torch.no_grad()!
            pred = likelihood(model(Xtest))
            preds.append(pred.mean.unsqueeze(1))
        return torch.cat(preds, dim=1)

