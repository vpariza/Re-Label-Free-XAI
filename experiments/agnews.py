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
sys.path.append(os.path.abspath('../src/lfxai/explanations'))
sys.path.append(os.path.abspath('../src/lfxai/models'))
sys.path.append(os.path.abspath('../src/lfxai/utils'))
#################################

import argparse
import logging
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import unicodedata
import re
import torch
# from captum.attr import GradientShap, IntegratedGradients, Saliency, LayerIntegratedGradients
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
import torchtext

from lfxai.explanations.examples_text import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.models.text import AGNewsAE
# from lfxai.explanations.features_text import attribute_auxiliary
from lfxai.utils.datasets import AG_NEWS_Tensors
# from lfxai.utils.feature_attribution import generate_tseries_masks
from lfxai.utils.metrics import similarity_rates

from tokenizers import BertWordPieceTokenizer


sos_token = '[CLS] '
eos_token = ' [SEP]'

def train_tokenizer(train_file):
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )

    # train BERT tokenizer
    tokenizer.train(
        train_file,
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        limit_alphabet=1000,
        wordpieces_prefix="##"
    )
    return tokenizer


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepareSentences(dataset, max_sentence_length):
    print("Reading lines...")
    
    sentences = []
    labels = []

    for (label_, text_) in iter(dataset):
        n_text_ = normalizeString(text_)
        n_text_ = sos_token + n_text_ + eos_token
        if len(n_text_) <= max_sentence_length:
          sentences.append(n_text_)
          labels.append(torch.tensor(label_ - 1)) # start with 0 as label instead of 1

    return (sentences, labels)


def filterData(data_pair, n_samples_per_class=2500):
    lbl_counter = {0:0, 1:0, 2:0, 3:0}
    sentences = data_pair[0]
    labels = data_pair[1]

    filtered_sentences = []
    filtered_labels = []

    for i in range(len(sentences)):
        if lbl_counter[labels[i].item()] < n_samples_per_class:
            lbl_counter[labels[i].item()] += 1
            filtered_sentences.append(sentences[i])
            filtered_labels.append(labels[i])

    return (filtered_sentences, filtered_labels)


def prepareData(data_dir, max_sentence_length=75):
    dataset = torchtext.datasets.AG_NEWS(root=data_dir, split='train')
    train_pair = prepareSentences(dataset, max_sentence_length)

    dataset = torchtext.datasets.AG_NEWS(root=data_dir, split='test')
    test_pair = prepareSentences(dataset, max_sentence_length)

    # Shuffle train data
    temp = list(zip(train_pair[0], train_pair[1]))
    random.shuffle(temp)
    st, lbl = zip(*temp)
    train_pair = (list(st), list(lbl))

    # Shuffle test data
    temp = list(zip(test_pair[0], test_pair[1]))
    random.shuffle(temp)
    st, lbl = zip(*temp)
    test_pair = (list(st), list(lbl))

    train_pair = filterData(train_pair, 2500)
    test_pair = filterData(test_pair, 500)

    train_sentences = train_pair[0] + test_pair[0]
    train_file = '.train_sentences.txt'

    with open(train_file, 'w+') as f:
        for items in train_sentences:
            f.write('%s\n' %items)
        print("File written successfully")

    return train_tokenizer(train_file), train_pair, test_pair


def consistency_feature_importance():
    pass


def consistency_example_importance(
    random_seed: int = 1,
    batch_size: int = 1,
    dim_latent: int = 128,
    n_epochs: int = 8,
    subtrain_size: int = 200,
    checkpoint_interval: int = 1,
    use_saved_dataset: bool = False,
    load_models: bool = True,
    load_metrics: bool = False,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_dir = Path.cwd() / "data/agnews"
    save_dir = Path.cwd() / "results/agnews/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load dataset
    max_sentences_length = 200 # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_dir = Path.cwd() / "data/agnews"
    save_dir = Path.cwd() / "results/agnews/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)

    if load_metrics is not True:
        # Load dataset
        max_sentences_length = 200
        MAX_LEN = 64

        if use_saved_dataset:
            print("Loading existing saved dataset")
            train_dataset = torch.load('results/agnews/consistency_examples/train_dataset.pt')
            test_dataset = torch.load('results/agnews/consistency_examples/test_dataset.pt')
        else:
            tokenizer, train_pair, test_pair = prepareData(data_dir, max_sentences_length)

            train_dataset = AG_NEWS_Tensors(
                sentences=train_pair[0], labels=train_pair[1], 
                tokenizer=tokenizer, max_len=MAX_LEN, device=device)
            test_dataset = AG_NEWS_Tensors(
                sentences=test_pair[0], labels=test_pair[1], 
                tokenizer=tokenizer, max_len=MAX_LEN, device=device)

        MAX_LEN = 64

        if use_saved_dataset:
            print("Loading existing saved dataset")
            train_dataset = torch.load('results/agnews/consistency_examples/train_dataset.pt')
            test_dataset = torch.load('results/agnews/consistency_examples/test_dataset.pt')
        else:
            tokenizer, train_pair, test_pair = prepareData(data_dir, max_sentences_length)

            train_dataset = AG_NEWS_Tensors(
                sentences=train_pair[0], labels=train_pair[1], 
                tokenizer=tokenizer, max_len=MAX_LEN, device=device)
            test_dataset = AG_NEWS_Tensors(
                sentences=test_pair[0], labels=test_pair[1], 
                tokenizer=tokenizer, max_len=MAX_LEN, device=device)


        train_loader = DataLoader(train_dataset, batch_size=None)
        test_loader = DataLoader(test_dataset, batch_size=None)
        vocab_size = train_dataset.tokenizer.get_vocab_size()

        # Train the denoising autoencoder
        logging.info("Fitting autoencoder")
        autoencoder = AGNewsAE(device, MAX_LEN, vocab_size, dim_latent)

        name = autoencoder.name
        model_loaded = False
        if load_models == True:
            if (save_dir / (name + ".pt")).is_file():
                logging.info('Loading the pretrained model from: {}'.format((save_dir / (name + ".pt"))))
                model_loaded = True
            else:
                logging.info('Cannot load a model from: {}'.format((save_dir / (name + ".pt"))))

        if model_loaded == False:
            # Train the denoising autoencoder
            logging.info('Training the model from scratch.')
            logging.info(f"Now fitting {name}")
            autoencoder.fit(
            device,
            train_loader,
            test_loader,
            save_dir,
            n_epochs,
            checkpoint_interval=checkpoint_interval,
            )
        
        autoencoder.load_state_dict(
            torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
        )

        # Prepare subset loaders for example-based explanation methods
        y_train = torch.tensor([train_dataset[k][1] for k in range(len(train_dataset))])
        idx_subtrain = [
            torch.nonzero(y_train == (n % 2))[n // 2].item() for n in range(subtrain_size)
        ]
        idx_subtest = torch.randperm(len(test_dataset))[:subtrain_size]
        train_subset = Subset(train_dataset, idx_subtrain)
        test_subset = Subset(test_dataset, idx_subtest)
        subtrain_loader = DataLoader(train_subset, batch_size=None)
        subtest_loader = DataLoader(test_subset, batch_size=None)

        labels_subtrain = torch.stack([label for _, label in subtrain_loader])
        labels_subtest = torch.stack([label for _, label in subtest_loader])
        recursion_depth = 100
        train_sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=recursion_depth * batch_size
        )
        train_loader_replacement = DataLoader(
            train_dataset, batch_size=None, sampler=train_sampler
        )

        if len(autoencoder.checkpoints_files) == 0:
            for epoch in range(n_epochs):
                n_checkpoint = 1 + epoch // checkpoint_interval
                path_to_checkpoint = (
                    save_dir / f"{autoencoder.name}_checkpoint{n_checkpoint}.pt"
                )
                autoencoder.checkpoints_files.append(path_to_checkpoint)

        # Fitting explainers, computing the metric and saving everything
        autoencoder.train().to(device)
        nll_loss = torch.nn.NLLLoss()
        explainer_list = [
            InfluenceFunctions(autoencoder, nll_loss, save_dir / "if_grads"),
            TracIn(autoencoder, nll_loss, save_dir / "tracin_grads"),
            SimplEx(autoencoder, nll_loss),
            NearestNeighbours(autoencoder, nll_loss),
        ]
        results_list = []
        # n_top_list = [1, 2, 5, 10, 20, 30, 40, 50, 100]
        frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
        n_top_list = [int(frac * len(idx_subtrain)) for frac in frac_list]
        for explainer in explainer_list:
            logging.info(f"Now fitting {explainer} explainer")
            if isinstance(explainer, InfluenceFunctions):
                with torch.backends.cudnn.flags(enabled=False):
                    attribution = explainer.attribute_loader(
                        device,
                        subtrain_loader,
                        subtest_loader,
                        train_loader_replacement=train_loader_replacement,
                        recursion_depth=recursion_depth,
                    )
            else:
                attribution = explainer.attribute_loader(
                    device, subtrain_loader, subtest_loader
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
    
    if (save_dir / "metrics.csv").is_file():
        logging.info('Loading the metrics from: {}'.format((save_dir / "metrics.csv")))
        results_df = pd.read_csv(save_dir / "metrics.csv")
    else:
        logging.info('Cannot load a metrics from: {}'.format((save_dir / "metrics.csv")))

    sns.lineplot(
        data=results_df,
        x="% Examples Selected",
        y="Similarity Rate",
        hue="Explainer",
        style="Type of Examples",
        palette="colorblind",
    )
    plt.savefig(save_dir / "agnews_similarity_rates.pdf")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="consistency_features")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=200)
    parser.add_argument("--use_saved_dataset", type=bool, default=False)
    args = parser.parse_args()
    if args.name == "consistency_features":
        consistency_feature_importance(
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            dim_latent=args.dim_latent,
        )
    elif args.name == "consistency_examples":
        consistency_example_importance(
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            dim_latent=args.dim_latent,
            subtrain_size=args.subset_size,
            checkpoint_interval=args.checkpoint_interval,
            use_saved_dataset=args.use_saved_dataset,
        )
    else:
        raise ValueError("Invalid experiment name.")
