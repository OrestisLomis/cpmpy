"""
Pseudo Boolean Competition (PB) Dataset

https://www.cril.univ-artois.fr/PB25/
"""

import lzma
import os
import pathlib
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import tarfile

from ....model import Model

from .._base import _Dataset


class CNFDataset(_Dataset): 
    """
    Access to local.
    """

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "OPT-LIN", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of local CNF instances.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): year (legacy).
            track (str): Track name specifying which subset of the competition instances to load (default="OPT-LIN").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        # Check requested dataset
        if not year:
            raise ValueError("Year must be specified")
        if not track:
            raise ValueError("Track must be specified, e.g. exact-weighted, exact-unweighted, ...")

        dataset_dir = self.root / str(year) / track
        
        print(dataset_dir)

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".cnf"
        )

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def metadata(self, file) -> dict:
        # Add the author to the metadata
        return super().metadata(file) | {'author': str(file).split(os.sep)[-1].split("_")[0],}
                

    def download(self):
        raise NotImplementedError("Dataset is not publicly downloadable")

    def open(self, instance: os.PathLike) -> callable:
        return lzma.open(instance, 'rt') if str(instance).endswith(".xz") else open(instance)

if __name__ == "__main__":
    dataset = CNFDataset(year=2024, track="DEC-LIN", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
