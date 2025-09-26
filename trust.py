from copy import deepcopy
import random 

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

from baselines.lame import kNN_affinity, rbf_affinity, linear_affinity, laplacian_optimization

class TRUST(nn.Module):
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
            loss = loss_normal -  0.1 * loss_mean

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

    def weighted_average_by_entropy(self, outputs_list):

        probs_list = [F.softmax(outputs, dim=-1) for outputs in outputs_list]
        entropies = torch.tensor([-torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean() for probs in probs_list])
        inv_entropy = 1.0 / entropies
        
        weights = inv_entropy / inv_entropy.sum()
        return weights

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        if self.args.experiment_name == 'traverse_permutation':
                    
            output_ewa = []
            theta_prim = []
            for exp in self.args.weight_list:
                model.train()

                # forward
                outputs = model(x, exp)
                # adapt
                loss = self.compute_loss_taskvector(outputs, x, exp)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                output_ewa.append(outputs)
                theta_prim.append(self.get_model_parameters(model))

                if self.args.mode_variation == 'parallel':
                    self.reset()

            entropy_wa = self.weighted_average_by_entropy(output_ewa)
            return (outputs, theta_prim, entropy_wa), loss
   
        else:
            model.train()

            # forward
            outputs = model(x)
            # adapt
            loss = self.compute_loss(outputs, x)
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
