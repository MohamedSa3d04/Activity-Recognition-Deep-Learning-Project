import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from data_utilites import get_frame_paths
from PIL import Image
import os, time

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
        class_index = self.categories_dct[target]



        return frame_tensor, torch.tensor(class_index, dtype=torch.long)
    
    
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

   
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 8)
        ) # Replace last layer with Specific New 8 output classes
    
    model.to(device)
    # Now let's freeze resinet layers except our new FC layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # unfreeze classifier
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    # Optimizer + Crietrion
    criterion = nn.CrossEntropyLoss()
    #(only trainable params)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-4)



    # Training Loop
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(data_loader):

            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Accuracy calculation
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Batch reporting
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] "
                    f"Batch [{batch_idx}/{len(data_loader)}] "
                    f"Loss: {loss.item():.4f}")

        # Epoch summary
        avg_loss = epoch_loss / len(data_loader)
        accuracy = correct / total * 100
        epoch_time = time.time() - start_time

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        print("-----------------------------\n")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            version_path = os.path.join(model_folder_path, f'V{epoch+1}.pth')
            torch.save(model.state_dict(), version_path)
            torch.save(model.state_dict(), last_version_path)
            print(f"Checkpoint saved: V{epoch+1}\n")
            

