import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Get current devices detalis
def check_device():
    print('torch: version', torch.__version__)
    # Check for availability of CUDA (GPU)
    if torch.cuda.is_available():
        print("CUDA is available.")
        # Get the number of GPU devices
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")

        # Print details for each CUDA device
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Get the name of the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")


def preprocess_images(image_level=True):
    # Crop the image to fit resinet input dim
    if image_level:
        preprocessor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        # already croped box. just resize it
        preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Let's return the resinet model with the preprocessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    
    resnet = models.resnet50(pretrained=True)

    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove final fc
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return preprocessor, feature_extractor, device


    

