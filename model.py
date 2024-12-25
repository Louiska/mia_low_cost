from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch
from torch.nn import Module

# load the data from a file
def get_model(architecture: str = "resnet18", pretrained_model: str = None) -> Module:
    """
    Args:
        name (str, optional): File name of model that's supposed to be loaded. If "", 
        only loads pretrained weights. Defaults to "".

    Returns:
        (Module): The loaded model
    """
    if architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(2048, 44)
    elif architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(512, 44)
    else:
        print("Architecture not supported, try resnet18 or resnet50")
        exit(1)
    if pretrained_model != None:
        model.load_state_dict(torch.load(f"out/models/{pretrained_model}.pt", map_location="cpu", weights_only=True))
    
    return model
