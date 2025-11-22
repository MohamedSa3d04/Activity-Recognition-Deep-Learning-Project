import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_utilites import get_frame_paths


class Data_Loader_BL1(Dataset):
    def __init__(self, main_path):
        super().__init__()
        frames_paths_tragets = get_frame_paths(main_path)
        print(len(frames_paths_tragets))


