import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import math
import sys
import time
import random
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import trust
import baselines.eata
import baselines.sar
import utils.utils.my_datasets as my_datasets
from utils.utils.sam import SAM

def import_abspy(name="models", path="classification/"):
    """Dynamically imports a module from a given path."""
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path), f"Invalid directory: {path}"
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

def argparser():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser("VMAMBA Test Time Adaptation")

    # Dataset
    parser.add_argument('--data_dir', type=str, default='utils/data', help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['cifar10c', 'cifar100c', 'imagenetc', 'imagenetsketch', 'imagenetv2', 'imagenetr', 'pacs'], help='Dataset to use')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--severity', type=int, default=5, help='Severity level')

    # Training settings
    parser.add_argument('--backbone', type=str, choices=['vmamba'], default='vmamba', help='Backbone model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--steps', type=int, help='Step size for adaptation')

    # Evaluation settings
    parser.add_argument('--adapt', type=str, choices=['True', 'False'], help='Enable or disable adapt mode')
    parser.add_argument('--episodic', type=str, choices=['True', 'False'], help='Reset model after each batch')
    parser.add_argument('--mode_variation', type=str, choices=['parallel', 'sequential'], help='Reset model after each batch')
    parser.add_argument('--model_merging', type=str, choices=['slerp', 'weight_averaging'], help='Reset model after each batch')
    parser.add_argument('--loss_choice', type=str, choices=['softmax_entropy', 'softmax_mean_entropy', 'pseudo_labeling'], help='Loss function')
    parser.add_argument('--adapted_parameter', type=str, choices=['bn', 'ss2d'], help='Adapted parameter')

    parser.add_argument('--experiment_name', type=str, choices=['source_only', 'tent', 'ssm', 'traverse_permutation', 'augmentation',
                                                                "test_traverses_0123", "test_traverses_0132", "test_traverses_0213", "test_traverses_0231",
                                                                "test_traverses_0312", "test_traverses_0321", "test_traverses_1023", "test_traverses_1032",
                                                                "test_traverses_1203", "test_traverses_1230", "test_traverses_1302", "test_traverses_1320",
                                                                "test_traverses_2013", "test_traverses_2031", "test_traverses_2103", "test_traverses_2130",
                                                                "test_traverses_2301", "test_traverses_2310", "test_traverses_3012", "test_traverses_3021",
                                                                "test_traverses_3102", "test_traverses_3120", "test_traverses_3201", "test_traverses_3210",
                                                                ], help='Experiment name')

    parser.add_argument('--corruptions_list', nargs='+', type=str, help='List of corruptions')
    
    parser.add_argument('--weight_list', nargs='+', type=str, help='List of corruptions')
    
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')

    return parser

def setup_logging(experiment_name, args):
    """Sets up logging for experiment results and logs argparse arguments."""
    log_filename = f"utils/results/{experiment_name}_{args.dataset}_{int(time.time())}.txt"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # Log all argparse arguments
    logging.info("Experiment Arguments:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

def set_seed(seed=42):
    """Fixes seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args, home_directory):
    """Loads the appropriate model based on arguments."""

    if args.backbone == 'vmamba':
        vmamba = import_abspy("vmamba", os.path.join(home_directory, "backbones/VMamba/classification/models"))
        model = vmamba.vmamba_tiny_s1l8(experiment_name=args.experiment_name, dataset=args.dataset).cuda()
        
        if args.dataset in {'imagenetc', 'imagenetsketch', 'imagenetv2', 'imagenetr'}:
            model.load_state_dict(torch.load(open(os.path.join(home_directory, "utils/ckpts/imagenet/ckpt_epoch_299_old.pth"), "rb"), map_location="cuda")['model'], strict=False)
        elif args.dataset == 'cifar10c':
            model.load_state_dict(torch.load(open(os.path.join(home_directory, "utils/ckpts/cifar10/ckpt_epoch_299_old.pth"), "rb"), map_location="cuda")['model'], strict=True)
        elif args.dataset == 'cifar100c':
            model.load_state_dict(torch.load(open(os.path.join(home_directory, "utils/ckpts/cifar100/ckpt_epoch_299_old.pth"), "rb"), map_location="cuda")['model'], strict=True)
        elif args.dataset == 'pacs':
            model.load_state_dict(torch.load(open(os.path.join(home_directory, "utils/ckpts/pacs/ckpt_epoch_299_old.pth"), "rb"), map_location="cuda")['model'], strict=True)

    else:
        raise ValueError(f"Invalid backbone: {args.backbone}")
    return model

def evaluate_model(model, device, test_loader, args, trust_model, corr_idx, corruption):
    """Evaluates the model before or after adaptation."""
    
    def log_batch_results(batch_idx, batch_accuracy, loss=None, log_every=50):

        if batch_idx % log_every != 0:
            return

        elapsed_time = time.time() - start_time
        batches_remaining = len(test_loader) - (batch_idx + 1)
        eta = elapsed_time / (batch_idx + 1) * batches_remaining
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

        if loss is None:
            message = f"Source only: [{batch_idx}/{len(test_loader)}] | eta {eta_str} | Accuracy {100.0 * batch_accuracy:.2f}%"
        else:
            message = f"Test: [{batch_idx}/{len(test_loader)}] | eta {eta_str} | Accuracy {100.0 * batch_accuracy:.2f}% | Loss {loss:.4f}"

        logging.info(message)

    def log_final_results(overall_accuracy):
        """Logs final accuracy results."""
        message = (f"Overall Accuracy: {overall_accuracy:.2f}%  |  "
                   f"Corruption: {corruption} [{corr_idx}/{len(args.corruptions_list)}]\n"
                   "------------------------------------------------------------------------------------------------------------------\n")
        logging.info(message)

    def log_entropy_results(mean_entropy, std_entropy):
        """Logs mean entropy and standard deviation."""
        message = f"Mean Entropy: {mean_entropy:.4f} | Std Entropy: {std_entropy:.4f}"
        logging.info(message)

    def entropy_from_logits(logits):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy

    # weights computation
    def mean_state_dict(state_dicts):
        num_dicts = len(state_dicts)
        mean_state_dict = {key: sum(d[key] for d in state_dicts) / num_dicts for key in state_dicts[0].keys()}
        return mean_state_dict

    def cosine_similarity(tensor1, tensor2):
        dot_product = torch.sum(tensor1 * tensor2)
        norm1 = torch.norm(tensor1)
        norm2 = torch.norm(tensor2)
        return dot_product / (norm1 * norm2 + 1e-8)

    def compute_theta(state_dict1, state_dict2):
        cos_theta = torch.mean(torch.tensor([
            cosine_similarity(state_dict1[key], state_dict2[key]) 
            for key in state_dict1.keys()
        ]))
        return torch.acos(cos_theta)

    def slerp_state_dict(state_dict1, state_dict2, c):
        theta = compute_theta(state_dict1, state_dict2)
        
        if theta == 0:  # If identical weights, return linear interpolation
            return {key: (1 - c) * state_dict1[key] + c * state_dict2[key] for key in state_dict1.keys()}
        
        sin_theta = torch.sin(theta)
        
        state_dict_slerped = {
            key: (torch.sin((1 - c) * theta) / sin_theta) * state_dict1[key] +
                (torch.sin(c * theta) / sin_theta) * state_dict2[key]
            for key in state_dict1.keys()
        }
        return state_dict_slerped

    def multi_slerp_state_dict(state_dicts, coefficients):
        assert len(state_dicts) == len(coefficients), "Number of models and coefficients must match"
        
        # Start with the first model
        state_dict_final = state_dicts[0]
        
        # Iteratively apply SLERP with the next model using its coefficient
        for i in range(1, len(state_dicts)):
            state_dict_final = slerp_state_dict(state_dict_final, state_dicts[i], coefficients[i])

        return state_dict_final

    total_correct, total_samples = 0, 0
    start_time = time.time()
    
    if args.adapt == 'False':
        with torch.no_grad():
            batch_mean_entropies = []
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = trust_model.evaluate(inputs)
                
                batch_mean_entropies.append(entropy_from_logits(outputs).mean().item())
                
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                batch_accuracy = batch_correct / labels.size(0)
                
                total_correct += batch_correct
                total_samples += labels.size(0)
                
                log_batch_results(batch_idx, batch_accuracy)
            
            mean_of_means = sum(batch_mean_entropies) / len(batch_mean_entropies)
            std_of_means = torch.std(torch.tensor(batch_mean_entropies)).item()
            log_entropy_results(mean_of_means, std_of_means)
            
    else:

        if args.experiment_name == 'traverse_permutation':
            trust_model.reset()
            
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if args.episodic == 'True':
                    trust_model.reset()
                
                (outputs, theta_prim, entropy_wa), loss = trust_model(inputs)  # Adaptation step

                if args.model_merging == 'slerp':
                    theta_final = multi_slerp_state_dict(theta_prim, entropy_wa)

                elif args.model_merging == 'weight_averaging':
                    theta_final = mean_state_dict(theta_prim)

                model.load_state_dict(theta_final, strict=False)

                outputs = trust_model.evaluate(inputs)  # Evaluation step
                
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                batch_accuracy = batch_correct / labels.size(0)
                
                total_correct += batch_correct
                total_samples += labels.size(0)
                
                log_batch_results(batch_idx, batch_accuracy, loss.item())

        elif args.experiment_name == 'augmentation':
            trust_model.reset()
            
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if args.episodic == 'True':
                    trust_model.reset()
                
                (outputs, theta_prim), loss = trust_model(inputs)  # Adaptation step

                if args.model_merging == 'weight_averaging':
                    theta_final = mean_state_dict(theta_prim)

                model.load_state_dict(theta_final, strict=False)

                outputs = trust_model.evaluate(inputs)  # Evaluation step
                
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                batch_accuracy = batch_correct / labels.size(0)
                
                total_correct += batch_correct
                total_samples += labels.size(0)
                
                log_batch_results(batch_idx, batch_accuracy, loss.item())

        else:
            trust_model.reset()
            
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if args.episodic == 'True':
                    trust_model.reset()
                
                outputs, loss = trust_model(inputs)  # Adaptation step
                outputs = trust_model.evaluate(inputs)  # Evaluation step
                
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                batch_accuracy = batch_correct / labels.size(0)
                
                total_correct += batch_correct
                total_samples += labels.size(0)
                
                log_batch_results(batch_idx, batch_accuracy, loss.item())
    
    overall_accuracy = 100.0 * total_correct / total_samples
    log_final_results(overall_accuracy)

def main():
    args = argparser().parse_args()
    setup_logging(args.experiment_name, args)

    set_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False 

    home_directory = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args, home_directory)

    ## TENT
    model = trust.configure_model(model, args.adapted_parameter)
    params, _ = trust.collect_params(model, args.adapted_parameter)
    optimizer = optim.Adam(params, lr=args.lr)
    trust_model = trust.TRUST(args, model, optimizer)

    ## ETA
    # model = eata.configure_model(model)
    # params, _ = eata.collect_params(model)
    # optimizer = torch.optim.Adam(params, args.lr)
    # trust_model = eata.EATA(model, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    
    ## SAR
    # model = sar.configure_model(model)
    # params, _ = sar.collect_params(model)
    # base_optimizer = torch.optim.Adam
    # optimizer = SAM(params, base_optimizer, lr=args.lr)
    # # optimizer = torch.optim.Adam(params, args.lr)
    # trust_model = sar.SAR(model, optimizer, margin_e0=args.sar_margin_e0)
    
    if args.dataset in {'imagenetc', 'cifar10c', 'cifar100c'}:
        for corr_idx, corruption in enumerate(args.corruptions_list):
            test_loader, _ = my_datasets.prepare_test_data(args, corruption)
            evaluate_model(model, device, test_loader, args, trust_model, corr_idx, corruption)
    else:
        test_loader, _ = my_datasets.prepare_test_data(args, corruption=None)
        evaluate_model(model, device, test_loader, args, trust_model, corr_idx=None, corruption=None)

if __name__ == "__main__":
    main()
