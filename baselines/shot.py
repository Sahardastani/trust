from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import matplotlib.pyplot as plt
import numpy as np


# from mamba_ssm.ops.triton.layer_norm import RMSNorm

class SHOT(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, args, model, optimizer):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        assert args.steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

        self.beta = args.beta

    def forward(self, x):

        for _ in range(self.args.steps):
            outputs, loss = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs, loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
            self.model_state, self.optimizer_state)

    def evaluate(self, x):
        """Evaluate the model on a batch of data without adaptation."""
        if self.args.experiment_name == 'traverse_permutation':
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x, 'test_traverses_0123')
            return outputs
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
            return outputs

    def compute_loss(self, outputs, x):
        if self.args.loss_choice == 'softmax_entropy':
            loss = softmax_entropy(outputs).mean(0)

        elif self.args.loss_choice == 'softmax_mean_entropy':
            loss_normal = softmax_entropy(outputs).mean(0)
            loss_mean = softmax_mean_entropy(outputs).mean(0)
            loss = loss_normal - 0.1 * loss_mean

        elif self.args.loss_choice == 'pseudo_labeling':
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
                probabilities = self.softmax(output)
                pseudo_labels = torch.argmax(probabilities, dim=1)
            self.model.train()
            loss = self.criterion(outputs, pseudo_labels)

        return loss

    def compute_loss_taskvector(self, outputs, x, exp):
        if self.args.loss_choice == 'pseudo_labeling':
            self.model.eval()
            with torch.no_grad():
                output = self.model(x, exp)
                probabilities = self.softmax(output)
                pseudo_labels = torch.argmax(probabilities, dim=1)
            self.model.train()
            loss = self.criterion(outputs, pseudo_labels)

        return loss

    def get_model_parameters(self, model):
        """
        Extracts the parameters of a PyTorch model and saves them in a dictionary.
        """
        param_dict = {name: param.clone().detach() for name, param in model.named_parameters()}
        return param_dict

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """

        outputs, features = model(x)
        features = features.flatten(1)
        C = outputs.shape[1]

        # Step 1: mutual information: H + KL
        L_ent = softmax_entropy(outputs).mean() # Entropy loss
        L_div = softmax_mean_entropy(outputs) # KL divergence

        # Step 2: DeepCluster
        # 2.1. Initialize centroids
        preds = outputs.softmax(1)
        num = preds.T @ features
        den = preds.sum(0).unsqueeze(1)
        centroids = num / (den + 1e-8)

        # 2.2 Distance-based pseudolabels
        centroids_norm = nn.functional.normalize(centroids, p=2, dim=1)
        features_norm = nn.functional.normalize(features, p=2, dim=1)
        dists = torch.cdist(features_norm, centroids_norm, p=2)
        plabels = dists.argmin(dim=1)

        # 2.3 Update centroids and pseudolabels
        one_hot = nn.functional.one_hot(plabels, num_classes=C).float()
        num = one_hot.T @ features
        den = one_hot.sum(0).unsqueeze(1)
        centroids = num / (den + 1e-8)
        centroids_norm = nn.functional.normalize(centroids, p=2, dim=1)
        dists = torch.cdist(features_norm, centroids_norm, p=2)
        plabels = dists.argmin(dim=1)
        y = nn.functional.one_hot(plabels, num_classes=C).float()

        # 2.4 Crossentropy with pseudolabels
        L_ce = nn.CrossEntropyLoss()(y, outputs)

        # Step 3: compute final loss
        loss = L_ent + L_div + self.beta * L_ce

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return outputs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = (x + eps).softmax(1).mean(0)
    return -(x * torch.log(x + eps)).sum()


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model, adapted_parameter):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    
    if adapted_parameter == 'bn':
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    elif adapted_parameter == 'ss2d':
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
            if hasattr(m, 'op'):
                for np, p in m.named_parameters():
                    if ('out_norm' not in np) and ('op' in np):
                        params.append(p)
                        names.append(f"{nm}.{np}")

    else:
        raise Exception('Adapted parameter not found!')
    
    return params, names

def configure_model(model, adapted_parameter):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics

    if adapted_parameter == 'bn':
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    elif adapted_parameter == 'ss2d':
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if hasattr(m, 'op'):
                m.op.requires_grad_(True)

    else:
        raise Exception('Adapted parameter not found!')

    return model