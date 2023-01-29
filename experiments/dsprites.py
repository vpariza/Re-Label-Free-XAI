# TODO: REVIEW & UPDATE- This code does not necessarily need to be part of the file, 
# but it is useful to guarantee that this module will find all the required modules to
# execute properly
######### MY CODE ADDITION TO ADD THE CODE PATH #########
import sys
import os
# Add the lfxai package to the path
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../src/'))
sys.path.append(os.path.abspath('../src/lfxai/'))
sys.path.append(os.path.abspath('../src/lfxai/explanations/'))
sys.path.append(os.path.abspath('../src/lfxai/models/'))
sys.path.append(os.path.abspath('../src/lfxai/utils/'))

#################################

import argparse
import csv
import itertools
import logging
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap, IntegratedGradients
from explanations.features import attribute_individual_dim
from torch.utils.data import random_split

from lfxai.models.images import VAE, DecoderBurgess, EncoderBurgess
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.utils.datasets import DSprites
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    spearman_saliency,
)
from lfxai.utils.visualize import plot_vae_saliencies, vae_box_plots
from lfxai.models.attr_priors import total_var_prior_attr

def disvae_feature_importance(
    random_seed: int = 1,
    batch_size: int = 500,
    n_plots: int = 10,
    n_runs: int = 5,
    dim_latent: int = 6,
    n_epochs: int = 100,
    beta_list: list = [1, 5, 10],
    test_split=0.1,
    reg_prior=None,
    attr_method_name='GradientShap',
    load_models=True,
    load_metrics=False,
    show_fig=False,
    override_metrics=False
) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dsprites
    W = 64
    img_size = (1, W, W)
    data_dir = Path.cwd() / "data/dsprites"
    dsprites_dataset = DSprites(str(data_dir))
    test_size = int(test_split * len(dsprites_dataset))
    train_size = len(dsprites_dataset) - test_size
    train_dataset, test_dataset = random_split(
        dsprites_dataset, [train_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create saving directory
    save_dir = None
    # if reg_prior is None then no prior is used
    if reg_prior is None:
        save_dir = Path.cwd() / "results/dsprites/vae/lat_dims_{}/{}/no_attr_prior".format(dim_latent, attr_method_name)
    else:
        save_dir = Path.cwd() / "results/dsprites/vae/lat_dims_{}/{}/pixel_attr_prior/{}".format(dim_latent, attr_method_name, reg_prior)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    metric_list = [
        pearson_saliency,
        spearman_saliency,
        cos_saliency,
        entropy_saliency,
        count_activated_neurons,
    ]
    metric_names = [
        "Pearson Correlation",
        "Spearman Correlation",
        "Cosine",
        "Entropy",
        "Active Neurons",
    ]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file() or override_metrics == True:
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, "w") as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=",", fieldnames=headers)
            dw.writeheader()
    # Available Attribution Methods to use
    attr_methods = {
        "GradientShap": GradientShap,
        "IntegratedGradients": IntegratedGradients
    }
    if load_metrics is not True:
        # Selected Attribution method for both the evaluation and the attribution prior
        attr_method = attr_methods[attr_method_name]
        for beta, loss, run in itertools.product(
            beta_list, loss_list, range(1, n_runs + 1)
        ):
            # Initialize vaes
            encoder = EncoderBurgess(img_size, dim_latent)
            decoder = DecoderBurgess(img_size, dim_latent)
            loss.beta = beta
            name = f"{str(loss)}-vae_beta{beta}_run{run}"
            model = None
            if reg_prior is None:
                # if reg_prior is None then no prior is used
                model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
            else:
                baseline_image = torch.zeros((1, 1, W, W), device=device)
                model = VAE(img_size, encoder, decoder, dim_latent, loss, 
                        attr_method=attr_method, 
                        baseline_input=baseline_image, 
                        attr_prior_loss_fn=total_var_prior_attr,
                        reg_prior=reg_prior,
                        name=name)
            logging.info(f"Working on {name}")
            # Keep track if model can be loaded
            model_loaded = False
            if load_models:
                if (save_dir / (name + ".pt")).is_file():
                    logging.info('Pretrained model loaded from: {}'.format((save_dir / (name + ".pt"))))
                    model_loaded = True 
                else:
                     logging.info('Cannot load pretrained module from: {}'.format((save_dir / (name + ".pt"))))
                
            if model_loaded == False:
                logging.info('Training the model from scratch.')
                logging.info(f"Now fitting {name}")
                model.fit(device, train_loader, test_loader, save_dir, n_epochs)
                logging.info('Model trained, saved and then loaded from: {}'.format((save_dir / (name + ".pt"))))
            model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
            # Compute test-set saliency and associated metrics
            baseline_image = torch.zeros((1, 1, W, W), device=device)
            gradshap = GradientShap(encoder.mu)
            attributions = attribute_individual_dim(
                encoder.mu, dim_latent, test_loader, device, gradshap, baseline_image
            )
            metrics = compute_metrics(attributions, metric_list)
            results_str = "\t".join(
                [f"{metric_names[k]} {metrics[k]:.2g}" for k in range(len(metric_list))]
            )
            logging.info(f"Model {name} \t {results_str}")

            # Save the metrics
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow([str(loss), beta] + metrics)

            # Plot a couple of examples
            plot_idx = [n for n in range(n_plots)]
            images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
            fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
            fig.savefig(save_dir / f"{name}.pdf")
            if show_fig:
                plt.show()
            plt.close(fig)
    else:
        logging.info('Using existing metrics to build figures.')
    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / "metric_box_plots.pdf")
    if show_fig:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--attr_method_name", type=str, default='GradientShap')
    parser.add_argument("--reg_prior", type=float, default=None)
    parser.add_argument("--load_models", action='store_true')
    parser.add_argument("--load_metrics", action='store_true')
    args = parser.parse_args()
    logging.info('Experiment Arguments')
    logging.info(str(args))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    disvae_feature_importance(
        n_runs=args.n_runs, batch_size=args.batch_size, 
                                random_seed=args.seed,
                                reg_prior=args.reg_prior,
                                attr_method_name=args.attr_method_name,
                                load_models=args.load_models, 
                                load_metrics=args.load_metrics
    )
    end_time = time.time()
    logging.info(f"Execution time: {(end_time - start_time):6.5f}s")
