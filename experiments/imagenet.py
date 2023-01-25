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

import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms
from torchvision.transforms import GaussianBlur, ToTensor
from lfxai.utils.datasets import TinyImageNet

from lfxai.explanations.examples import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.explanations.features import attribute_auxiliary, attribute_individual_dim
from lfxai.models.images import (
    AutoEncoderTinyImageNet,
    DecoderTinyImageNet,
    EncoderTinyImageNet,
)
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.models.pretext import Identity, Mask, RandomNoise
from lfxai.utils.datasets import MaskedMNIST
from lfxai.utils.feature_attribution import generate_masks
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    similarity_rates,
    spearman_saliency,
)
from lfxai.utils.visualize import (
    correlation_latex_table,
    plot_pretext_saliencies,
    plot_pretext_top_example,
    plot_vae_saliencies,
    vae_box_plots,
)


def consistency_feature_importance(
     random_seed: int = 1,
    batch_size: int = 1000,
    dim_latent: int = 4,
    n_epochs: int = 100,
    subtrain_size: int = 1000,
    subset_class: int = None,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if subset_class is not None:
        no_classes = subset_class
    else:
        no_classes = 200
    logging.info(f"Running for {no_classes} classes.")
    W = 64  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]
    perturbation = GaussianBlur(21, sigma=5).to(device)
    # Load Imagenet
    data_dir = Path.cwd() / "data/tinyimagenet"
    train_dataset = TinyImageNet(data_dir, train=True, download=True, subset_class=subset_class)
    test_dataset = TinyImageNet(data_dir, train=False, download=True, subset_class=subset_class, class_list=train_dataset._classes)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

       # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderTinyImageNet(encoded_space_dim=dim_latent)
    decoder = DecoderTinyImageNet(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderTinyImageNet(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)

       # Train the denoising autoencoder
    path =  "results/imagenet/consistency_features"+str(subset_class)+"_classes"
    save_dir = Path.cwd() / path
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )
   
    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Saliency": Saliency,
        "Random": None,
    }
    results_data = []

    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(
                encoder, test_loader, device, attr_method(encoder), perturbation
            )
        else:
            np.random.seed(random_seed)
            attr = np.random.randn(len(test_dataset), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(
                f"Perturbing {pert_percentage}% of the features with {method_name}"
            )
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[
                    batch_id * batch_size : batch_id * batch_size + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask*images + (1-mask)*perturbation(images)
                pert_reps = encoder(images)
                rep_shift = torch.mean(
                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
                ).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    logging.info("Saving the plot")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% Perturbed Pixels", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "imagenet_consistency_features.pdf")
    plt.close() 


def consistency_examples(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 16,
    n_epochs: int = 100,
    subtrain_size: int = 1000,
    subset_class: int = None,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if subset_class is not None:
        no_classes = subset_class
    else:
        no_classes = 200
    
    # Load MNIST
    data_dir = Path.cwd() / "data/tinyimagenet"
    train_dataset = TinyImageNet(data_dir, train=True, download=True, subset_class=subset_class)
    test_dataset = TinyImageNet(data_dir, train=False, download=True, subset_class=subset_class, class_list=train_dataset._classes)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderTinyImageNet(encoded_space_dim=dim_latent)
    decoder = DecoderTinyImageNet(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderTinyImageNet(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # Train the denoising autoencoder
    logging.info("Now fitting autoencoder")
    save_dir = Path.cwd() / "results/imagenet/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(
        device, train_loader, test_loader, save_dir, n_epochs, checkpoint_interval=10
    )
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )
    autoencoder.train().to(device)

    idx_subtrain = [
        torch.nonzero(train_dataset.targets == (n % no_classes))[n // no_classes].item()
        for n in range(subtrain_size)
    ]
    idx_subtest = [
        torch.nonzero(test_dataset.targets == (n % no_classes))[n // no_classes].item()
        for n in range(subtrain_size)
    ]
    train_subset = Subset(train_dataset, idx_subtrain)
    test_subset = Subset(test_dataset, idx_subtest)
    subtrain_loader = DataLoader(train_subset)
    subtest_loader = DataLoader(test_subset)
    labels_subtrain = torch.cat([label for _, label in subtrain_loader])
    labels_subtest = torch.cat([label for _, label in subtest_loader])

    # Create a training set sampler with replacement for computing influence functions
    recursion_depth = 100
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(
        train_dataset, batch_size, sampler=train_sampler
    )

    # Fitting explainers, computing the metric and saving everything
    mse_loss = torch.nn.MSELoss()
    # TODO: Inform the others for the fix in the NearestNeighbours instantiation
    explainer_list = [
        InfluenceFunctions(autoencoder, mse_loss, save_dir / "if_grads"),
        TracIn(autoencoder, mse_loss, save_dir / "tracin_grads"),
        SimplEx(autoencoder, mse_loss),
        NearestNeighbours(model=autoencoder, loss_f=mse_loss),
    ]
    frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    n_top_list = [int(frac * len(idx_subtrain)) for frac in frac_list]
    results_list = []
    for explainer in explainer_list:
        logging.info(f"Now fitting {explainer} explainer")
        attribution = explainer.attribute_loader(
            device,
            subtrain_loader,
            subtest_loader,
            train_loader_replacement=train_loader_replacement,
            recursion_depth=recursion_depth,
        )
        autoencoder.load_state_dict(
            torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
        )
        sim_most, sim_least = similarity_rates(
            attribution, labels_subtrain, labels_subtest, n_top_list
        )
        results_list += [
            [str(explainer), "Most Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_most)
        ]
        results_list += [
            [str(explainer), "Least Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_least)
        ]
    results_df = pd.DataFrame(
        results_list,
        columns=[
            "Explainer",
            "Type of Examples",
            "% Examples Selected",
            "Similarity Rate",
        ],
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")
    sns.lineplot(
        data=results_df,
        x="% Examples Selected",
        y="Similarity Rate",
        hue="Explainer",
        style="Type of Examples",
        palette="colorblind",
    )
    plt.savefig(save_dir / "similarity_rates.pdf")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="consistency_examples")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--subset_class", type=int, default=20)
    args = parser.parse_args()

    if args.name == "consistency_features":
        consistency_feature_importance(
            batch_size=args.batch_size, random_seed=args.random_seed, subset_class=args.subset_class)
    elif args.name == "consistency_examples":
        consistency_examples(batch_size=args.batch_size, random_seed=args.random_seed, subset_class=args.subset_class)
    else:
        raise ValueError("Invalid experiment name")