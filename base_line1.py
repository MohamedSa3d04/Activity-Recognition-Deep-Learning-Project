import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_utilites import get_frame_paths


class Data_Loader_BL1(Dataset):
    def __init__(self):
        super().__init__()
        all_paths = 0


