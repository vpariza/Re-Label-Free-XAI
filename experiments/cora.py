"""
This file is an entry point for calling label-free importance attribution
methods for the Graph Cora Dataset.
Usage: python -m cora --name consistency_features.
"""

import sys
import os
# Add the lfxai package to the path
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../src/'))
sys.path.append(os.path.abspath('../src/lfxai/'))
sys.path.append(os.path.abspath('../src/lfxai/explanations'))
sys.path.append(os.path.abspath('../src/lfxai/models'))
sys.path.append(os.path.abspath('../src/lfxai/utils'))
#################################
import time
import argparse
import csv
import itertools
import logging
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from torch import optim

from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms
from torchvision.transforms import GaussianBlur, ToTensor
from vision_tinyimagenet import TinyImageNet

import torch.nn as nn
from torch_geometric.nn import  to_captum_model

from lfxai.models.graph import (
   EncoderCora,
   DecoderCora,
   GraphAutoEncoderCora
)


from lfxai.utils.visualize import (
    correlation_latex_table,
    plot_pretext_saliencies,
    plot_pretext_top_example,
    plot_vae_saliencies,
    vae_box_plots,
)

from lfxai.utils.gae_utils import (
   new_load_data, 
   get_features_adj,
   get_roc_score,
   mask_test_edges,
   preprocess_graph
)

def generate_mask(imp_nodes, perc, adj_norm, is_random=False):
    """
     Generates mask by masking the edges between nodes in 
     such a way as to isolate them.
     imp_nodes: The importance attribution score returned.
     perc: The % of nodes to isolated
     adj_norm: Normalised adjacency matrix
     is_random: if True, then we randomly isolate nodes.

     Returns
     mask: An array of 1s, zeroed out where imp nodes are located by perc.
     adj_sub: Masked adjacency matrix

    """
    n_nodes= int(imp_nodes.size()[0]*perc/100.0)

    if is_random:
       indices = random.sample(range(0, imp_nodes.size()[0]), n_nodes)
    else:
        _ , indices = torch.topk(imp_nodes, k=n_nodes)
    mask = torch.ones(imp_nodes.size())
    mask[indices] = 0
    adj_dense = adj_norm.to_dense()
    # make those nodes isolated
    for idx in indices:
        adj_dense[idx,:] = 0
        adj_dense[:,idx] = 0
    adj_sub = adj_dense.to_sparse()
    return mask, adj_sub


def consistency_feature_importance(
     random_seed: int = 1,
    batch_size: int = 1000,
    dim_latent: int = 4,
    n_epochs: int = 3,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    dataset = 'Cora'
    data = new_load_data(dataset) # 
    features, adj = get_features_adj(data)


    # # Preprocess the data
    n_nodes, feat_dim = features.shape
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train   
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = torch.Tensor([(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    
    # Initialize the model
    encoder = EncoderCora(input_feat_dim=feat_dim)
    decoder = DecoderCora()
    autoencoder = GraphAutoEncoderCora(encoder, decoder)
    encoder.to(device)
    decoder.to(device)
  
    # Train the Graph vae
    save_dir = Path.cwd() / "results/cora/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)
    
 
    autoencoder.fit(device, features,adj_norm, adj_label, n_nodes, norm, pos_weight,
     adj_orig,val_edges, val_edges_false, test_edges, test_edges_false, save_dir, n_epochs)
    
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )
    
    # Node explainability
    model = encoder
    captum_model = to_captum_model(model.mu, mask_type='node')

    methods = {
        "Integrated Gradients": IntegratedGradients(captum_model),
        "Saliency": Saliency(captum_model),
        "Random": None,
    }

    orig_e_mu, _ = encoder(features, adj_norm)
    f_e_mu_rows = torch.mean(orig_e_mu, dim=0)

    # Reference Node calculated as per - https://arxiv.org/pdf/1903.03894.pdf Appendix Section A
    # find the node whose embedding is as close to that of mean.
    target_idx = torch.argmin(abs(orig_e_mu-f_e_mu_rows), dim=1)

    results_data = []
    adj_norm_removed = None
    for method_name, method_obj in methods.items():
        if "Integrated Gradients" in method_name:
                ig_attr_node = method_obj.attribute(features.unsqueeze(0), target=target_idx,
                                            additional_forward_args=(adj_norm),
                                            internal_batch_size=1)
                ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
                ig_attr_node /= ig_attr_node.max()
        elif "Saliency" in method_name:
                ig_attr_node = method_obj.attribute(features.unsqueeze(0), target=target_idx,
                                            additional_forward_args=(adj_norm)
                                                )
                ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
                ig_attr_node /= ig_attr_node.max()
        else:
                ig_attr_node = torch.ones(features.size()[0]) 

        for remove_nodes_perc in [5, 10, 20, 50, 80, 100]:
            # Visualize absolute values of attributions:

            if method_name == "random":
                mask, adj_perturbed = generate_mask(ig_attr_node, remove_nodes_perc, adj_norm, is_random=True)
            else:
                mask, adj_perturbed = generate_mask(ig_attr_node, remove_nodes_perc, adj_norm)
            pert_rep, _ = encoder(features, adj_perturbed)
            shift = torch.mean(
                        torch.sum((orig_e_mu - pert_rep) ** 2, dim=-1)
                    ).item()
       
            results_data.append([method_name, remove_nodes_perc, shift])

    
    results_df = pd.DataFrame(
        results_data, columns=["Method", "Nodes Isolated (%)", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="Nodes Isolated (%)", y="Representation Shift", hue="Method"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "consistency_features.pdf")
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="consistency_features")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=1)
    args = parser.parse_args()

    if args.name == "consistency_features":
        consistency_feature_importance(
            batch_size=args.batch_size, random_seed=args.random_seed
        )        
    else:
        raise ValueError("Invalid experiment name")