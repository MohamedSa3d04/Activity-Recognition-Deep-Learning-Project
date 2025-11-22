import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from data_utilites import get_frame_paths
import cv2


class Data_Loader_BL1(Dataset):
    def __init__(self, main_path):
        super().__init__()
        self.frames_paths_tragets = get_frame_paths(main_path)
        self.categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
        }

        self.preprocessor = transforms.Compose(
            [
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    
    def __len__(self):
        return len(self.frames_paths_tragets)
    
    def __getitem__(self, index):
        # Get frame_path, target
        frame_path, target = self.frames_paths_tragets[index]
        

        # Read The Frame
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #To RGB
        frame_tensor = self.preprocessor(frame)

        # Read The Target
        class_ = self.categories_dct[target]
        target = ([0] * 8)
        target[class_] = 1


        return frame_tensor, torch.tensor(target)
    
    
def run(main_path):
    dataset = Data_Loader_BL1(main_path)
    data_loader = DataLoader(dataset, 30, shuffle=True) 
    for frame_tensor, target in data_loader:
        print(frame_tensor, target)
        break

    print(frame_tensor.shape, target.shape)


