import abc
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict, List
from tqdm import tqdm
import random
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import wget
from PIL import Image
from scipy.io.arff import loadarff
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive
from torchvision.datasets.vision import VisionDataset
from tokenizers import BertWordPieceTokenizer

"""
The code for DSprites is adapted from https://github.com/YannDubs/disentangling-vae/blob/master/utils/datasets.py
"""


DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"dsprites": "DSprites"}
DATASETS = list(DATASETS_DICT.keys())


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters:
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info(f"Downloading {str(type(self))} ...")
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset."""
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].
    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.
    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters:
    ----------
    root : string
        Root directory of dataset.
    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.
    """

    urls = {
        "train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    }
    files = {"train": "dsprite_train.npz"}
    lat_names = ("shape", "scale", "orientation", "posX", "posY")
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {
        "posX": np.array(
            [
                0.0,
                0.03225806,
                0.06451613,
                0.09677419,
                0.12903226,
                0.16129032,
                0.19354839,
                0.22580645,
                0.25806452,
                0.29032258,
                0.32258065,
                0.35483871,
                0.38709677,
                0.41935484,
                0.4516129,
                0.48387097,
                0.51612903,
                0.5483871,
                0.58064516,
                0.61290323,
                0.64516129,
                0.67741935,
                0.70967742,
                0.74193548,
                0.77419355,
                0.80645161,
                0.83870968,
                0.87096774,
                0.90322581,
                0.93548387,
                0.96774194,
                1.0,
            ]
        ),
        "posY": np.array(
            [
                0.0,
                0.03225806,
                0.06451613,
                0.09677419,
                0.12903226,
                0.16129032,
                0.19354839,
                0.22580645,
                0.25806452,
                0.29032258,
                0.32258065,
                0.35483871,
                0.38709677,
                0.41935484,
                0.4516129,
                0.48387097,
                0.51612903,
                0.5483871,
                0.58064516,
                0.61290323,
                0.64516129,
                0.67741935,
                0.70967742,
                0.74193548,
                0.77419355,
                0.80645161,
                0.83870968,
                0.87096774,
                0.90322581,
                0.93548387,
                0.96774194,
                1.0,
            ]
        ),
        "scale": np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "orientation": np.array(
            [
                0.0,
                0.16110732,
                0.32221463,
                0.48332195,
                0.64442926,
                0.80553658,
                0.96664389,
                1.12775121,
                1.28885852,
                1.44996584,
                1.61107316,
                1.77218047,
                1.93328779,
                2.0943951,
                2.25550242,
                2.41660973,
                2.57771705,
                2.73882436,
                2.89993168,
                3.061039,
                3.22214631,
                3.38325363,
                3.54436094,
                3.70546826,
                3.86657557,
                4.02768289,
                4.1887902,
                4.34989752,
                4.51100484,
                4.67211215,
                4.83321947,
                4.99432678,
                5.1554341,
                5.31654141,
                5.47764873,
                5.63875604,
                5.79986336,
                5.96097068,
                6.12207799,
                6.28318531,
            ]
        ),
        "shape": np.array([1.0, 2.0, 3.0]),
        "color": np.array([1.0]),
    }

    def __init__(self, root=os.path.join(DIR, "../data/dsprites/"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip["imgs"]
        self.lat_values = dataset_zip["latents_values"]

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(
            ["curl", "-L", type(self).urls["train"], "--output", self.train_data]
        )

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


class ECG5000(Dataset):
    def __init__(
        self,
        dir: pathlib.Path,
        train: bool = True,
        random_seed: int = 42,
        experiment: str = "features",
    ):
        if experiment not in ["features", "examples"]:
            raise ValueError("The experiment name is either features or examples.")
        self.dir = dir
        self.train = train
        self.random_seed = random_seed
        if not dir.exists():
            os.makedirs(dir)
            self.download()

        # Load the data and create a train/test set with split
        with open(self.dir / "ECG5000_TRAIN.arff") as f:
            data, _ = loadarff(f)
            total_df = pd.DataFrame(data)
        with open(self.dir / "ECG5000_TEST.arff") as f:
            data, _ = loadarff(f)
            total_df = total_df.append(pd.DataFrame(data))

        # Isolate the target column in the dataset
        label_normal = b"1"
        new_columns = list(total_df.columns)
        new_columns[-1] = "target"
        total_df.columns = new_columns

        if experiment == "features":
            # Split the dataset in normal and abnormal examples
            normal_df = total_df[total_df.target == label_normal].drop(
                labels="target", axis=1
            )
            anomaly_df = total_df[total_df.target != label_normal].drop(
                labels="target", axis=1
            )
            if self.train:
                df = normal_df
            else:
                df = anomaly_df
            labels = [int(self.train) for _ in range(len(df))]

        elif experiment == "examples":
            df = total_df.drop(labels="target", axis=1)
            labels = [0 if label == label_normal else 1 for label in total_df.target]

        else:
            raise ValueError("Invalid experiment name.")

        sequences = df.astype(np.float32).to_numpy().tolist()
        sequences = [
            torch.tensor(sequence).unsqueeze(1).float() for sequence in sequences
        ]
        self.sequences = sequences
        self.labels = labels
        self.n_seq, self.seq_len, self.n_features = torch.stack(sequences).shape

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def download(self):
        """Download the dataset."""
        url = "http://timeseriesclassification.com/Downloads/ECG5000.zip"
        logging.info("Downloading the ECG5000 Dataset.")
        data_zip = self.dir / "ECG5000.zip"
        wget.download(url, str(data_zip))
        with ZipFile(data_zip, "r") as zip_ref:
            zip_ref.extractall(self.dir)
        logging.info("Finished Downloading.")


class MaskedMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        masks: torch.Tensor = None,
    ):
        super().__init__(root, train=train, download=True)
        self.masks = masks

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        image = self.masks[index] * image
        return image, target


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair


# EXTENSIONS

class AG_NEWS_Tensors(Dataset):
    def __init__(
        self,
        device: torch.device,
        sentences: list,
        labels: list,
        tokenizer: BertWordPieceTokenizer,
        max_len: int = 64,
        random_seed: int = 42,
    ):
        self.max_len = max_len
        self.random_seed = random_seed
        self.tokenizer = tokenizer
        self.sequences = []
        self.sentences = []
        self.labels = []
        self.device = device

        for i in range(len(sentences)):
            seq = torch.squeeze(torch.tensor(self.tokenizer.encode(sentences[i]).ids, device=self.device))
            if seq.size(0) < self.max_len:
                self.sentences.append(sentences[i])
                self.sequences.append(seq)
                self.labels.append(labels[i])
        print("Trimmed to %s sequences" % len(self.sequences))

        self.n_seq = len(self.sequences)
        

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


"""
torchvision does not have support for tiny imagenet dataset yet,
this implementation has been picked up from - https://github.com/towzeur/vision/commit/a67feb569361f440fd48ed492183de8bd8f6b585
"""

class TinyImageNet(VisionDataset):
    """`TinyImageNet <https://www.kaggle.com/c/tiny-imagenet>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    train_list = [
        ["train_data", "894532b79836a003b4effcf9ae778f8d"],
        ["train_targets", "2a2ab983ba40b23a79b293c30d894fa9"],
        ["train_bboxes", "613dd1e1ad67c6d24a11935472c0ba63"]
    ]

    test_list = [
        ["test_data", "81884071c408ec2ff7b45fbde81da748"],
        ["test_targets", "3b39162ddb25e2a2743791005ba0e0fa"],
        ["test_bboxes", "1de94c9b303983505c44e76803b396e8"]
    ]

    NUM_CLASSES = 200
    INPUT_SIZE = 64


    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            subset_class: int = None,
            class_list: list = None,
    ) -> None:

        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.split = 'train' if train else 'test'
        self.subset_class = subset_class
        self._classes = class_list

        self.TRAIN_SIZE = 500*self.subset_class if self.subset_class is not None else 100000
        self.TEST_SIZE = 50*self.subset_class if self.subset_class is not None else 10000
        self.base_dir = Path(root) / self.base_folder
        self.zip_file =  Path(root) / self.filename
        self.split_dir = self.base_dir / ('train' if train else 'val')
        self.npy_dir = self.base_dir / 'npy'

        # download zip file
        if download:
            self.download()
        if not self._check_zip_integrity():
            raise RuntimeError(f"`{self.filename}` not found or corrupted. You can use download=True to download it")
        if not self.base_dir.exists():
            print("Archive not extracted. Extracting...")
            extract_archive(self.zip_file, self.base_dir)

        self._load_meta()
        self._load_or_construct_files()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _load_or_construct_files(self) -> None:
        '''
        load if files exist, otherwise construct them
        numpy files for quick load and access
        '''
        self.data_file = self.npy_dir / f'{self.split}_data.npy'
        self.targets_file = self.npy_dir / f'{self.split}_targets.npy'
        self.bboxes_file = self.npy_dir / f'{self.split}_bboxes.npy'

        if self._check_integrity():
            print("Numpy files already constructed and verified")
            # load numpy files
            self.data = np.load(self.data_file)
            self.targets = np.load(self.targets_file)
            self.bboxes = np.load(self.bboxes_file, allow_pickle=True)
        else:
            print("Numpy files not found or corrupted. Constructing...")
            self._parse_dataset()

            # save for quick access:
            self.npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.data_file, self.data)
            np.save(self.targets_file, self.targets)
            np.save(self.bboxes_file, self.bboxes)

    def _load_meta(self) -> None:
        # _classes = [n02124075,...,n02504458]

        if self._classes == None:
            with (self.base_dir / 'wnids.txt').open() as file:
                self._classes = [x.strip() for x in file.readlines()]
                if self.subset_class is not None:
                    rand_idx = random.sample(range(len(self._classes)), self.subset_class)
                    self._classes = [self._classes[i] for i in rand_idx]

        self.class_to_idx = {name:i for i, name in enumerate(self._classes)}
        self.idx_to_class = {i:name for i, name in enumerate(self._classes)}

        # classes = ['Egyptian cat',...,'African elephant, Loxodonta africana']
        self.classes = [None] * len(self._classes)
        with (self.base_dir / 'words.txt').open() as file:
            for line in file:
                name, readable_name = line.rstrip().split('\t')
                if name in self.class_to_idx:
                    self.classes[self.class_to_idx[name]]=readable_name

    def _check_integrity(self) -> bool:
        split_list = self.train_list if self.train else self.test_list
        for filename, md5 in split_list:
            fpath = self.npy_dir / (filename + '.npy')
            if not check_integrity(fpath, md5):
                return False
        return True

    def _check_zip_integrity(self) -> bool:
        return check_integrity(self.zip_file, self.zip_md5)

    def download(self) -> None:
        if self._check_zip_integrity():
            print("Archive already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.zip_md5)

    def extra_repr(self) -> str:
        return f"Split: {self.split.capitalize()}"

    def _parse_image(self, path) -> np.ndarray:
        img = Image.open(path)
        np_img = np.array(img)
        assert np_img.ndim in (2, 3), f'Image dim shoud be 2 or 3, but is {np_img.ndim}'
        assert np_img.shape[:2] == (self.INPUT_SIZE,)*2, f'Illegal shape of {np_img.shape}'
        if np_img.ndim == 2:
            np_img = np.stack((np_img, ) * 3, axis=-1)
        return np_img

    def _parse_dataset(self):
        '''
        generates npy files from the original folder dataset
        '''
        print(f'Parsing {self.split} data...')
        samples = []
        iter = self._classes if self.train else range(1)
        for cls in tqdm(iter, desc=" outer", position=0):
            if self.train:
                boxes_file = self.split_dir / cls / (cls + '_boxes.txt')
            else:
                boxes_file = self.split_dir / 'val_annotations.txt'
            with boxes_file.open() as boxes_file:
                lines = boxes_file.readlines()

            for line in tqdm(lines, position=1, leave=False):
                if self.train:
                    filename, *bbox = line.rstrip().split('\t')
                    path = self.split_dir / cls / 'images' / filename
                else:
                    filename, cls, *bbox = line.rstrip().split('\t')
                    path = self.split_dir / 'images' / filename

                if cls in self._classes:
                    target = self.class_to_idx[cls]
                    bbox = map(int, bbox)
                    image = self._parse_image(path)
                    samples.append((image, target, bbox))

        image, target, bboxes = zip(*samples)
        self.data = np.stack(image)
        self.targets = torch.from_numpy(np.array(target))
        self.bboxes = np.array(bboxes)