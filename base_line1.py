import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from data_utilites import get_frame_paths
from PIL import Image
import os

from torchvision.models import resnet50

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
        frame = Image.open(frame_path).convert('RGB')
        frame_tensor = self.preprocessor(frame)

        # Read The Target
        class_ = self.categories_dct[target]
        target = [0] * 8
        target[class_] = 1


        return frame_tensor, torch.tensor(target)
    
    
def run(main_videos_path, models_path):

    # Let's dive in the training: 
    n_epochs = 25

    # - Loading the data
    dataset = Data_Loader_BL1(main_videos_path)
    data_loader = DataLoader(dataset, 50, shuffle=True) 

    # Folder to save Model Versions
    model_folder_path = os.path.join(models_path, 'BaseLine1')
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    # Set-Up The Training
    # device:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    
    # Model Fine-Tuined Resinet:

    # Check If there is a checkpoint or not
    model = resnet50(pretrained=True)
    last_version_path = os.path.join(model_folder_path, f'LastVersion.pth')
    if os.path.exists(last_version_path):
        print("CheckPoint Existed!")
        state_dict = torch.load(last_version_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 8)
        ) # Replace last layer with Specific New 8 output classes

    # Now let's freeze resinet layers except our new FC layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # unfreeze classifier
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    # Optimizer + Crietrion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



    # Training Loop
    model.train()
    for i in range(n_epochs):
        for index, (frame_tensor, target) in enumerate(data_loader):
            # get the output and calc the loss
            output = model(frame_tensor.to(device)) 
            loss = criterion(output, target.to(device))

            # zero the optimizer (For not aggeregration)
            optimizer.zero_grad()
            loss.backward() # Compute gradients
            optimizer.step() # Adjust learning weights

            # report loss
            running_loss += loss.item()
            if index % 20 == 0: # For each 25 batches Report loss
                last_loss = running_loss / 25 # loss per batch
                print(f'Epoch {i}, Batch {index} -> {last_loss}')
                running_loss = 0


         # For each 5 epchs save a copy version
        if i % 5 == 0:
            torch.save(model.state_dict(), f=os.path.join(model_folder_path, f'V{i}.pth'))
            torch.save(model.state_dict(), f=last_version_path) #Override Last Version
          

