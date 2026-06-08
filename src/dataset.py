"""Spatial transcriptomics dataset loaders with lazy UNI2-h feature loading via PyG."""

import glob
import logging
import os
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


class BaseSpatialDataset(Dataset, ABC):
    """
    Abstract Base Class for Spatial Transcriptomics Datasets.
    Implements Lazy Loading architecture to save RAM.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        self.file_paths: list[str] = []
        self._scan_files()

        if len(self.file_paths) == 0:
            logger.warning(f"No data files found in {root_dir}")
        else:
            logger.info(f"Found {len(self.file_paths)} data files in {root_dir}")

    @abstractmethod
    def _scan_files(self):
        """Implement logic to populate self.file_paths."""
        pass

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data | Batch:
        path = self.file_paths[idx]
        try:
            return self._load_one_file(path)
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            raise e

    @abstractmethod
    def _load_one_file(self, path: str) -> Data | Batch:
        """Specific logic to load a single file and convert to Data/Batch."""
        pass


class Her2stDataset(BaseSpatialDataset):
    """
    Her2st Dataset implementation with Lazy Loading.
    """

    def _scan_files(self):
        self.file_paths = sorted(glob.glob(os.path.join(self.root_dir, "*.pt")))

    def _load_one_file(self, path: str) -> Data | Batch:

        data_obj = torch.load(path, map_location="cpu", weights_only=False)

        sid = os.path.basename(path).replace(".pt", "")

        if isinstance(data_obj, list):
            if len(data_obj) > 0:
                batch = Batch.from_data_list(data_obj)
                batch.patient_id = sid
                return batch
            else:
                raise ValueError(f"File {path} contains empty list.")
        elif isinstance(data_obj, Data):
            data_obj.patient_id = sid
            return data_obj
        else:
            raise TypeError(f"Unknown data type in {path}: {type(data_obj)}")

    def get_patient_indices(self) -> dict[str, list[int]]:
        """
        Returns a dictionary mapping patient IDs (grouped by prefix) to dataset indices.
        Supports both HER2ST format ('A1.pt' -> pid='A') and
        cSCC/other format ('P2_ST_rep1.pt' -> pid='P2').
        """
        pm = {}
        for idx, path in enumerate(self.file_paths):
            filename = os.path.basename(path)
            sid = filename.replace(".pt", "")

            # Try to detect patient ID format
            if "_" in sid:
                # Multi-part filename: extract patient prefix before first '_ST' or '_rep'
                parts = sid.split("_")
                # Find patient part (e.g., 'P2', 'P10')
                pid = parts[0]
                for p in parts:
                    if p.startswith("P") and len(p) > 1 and p[1:].isdigit():
                        pid = p
                        break
            else:
                # Simple filename: 'A1' -> pid='A'
                pid = sid[0]

            if pid not in pm:
                pm[pid] = []
            pm[pid].append(idx)
        return pm
