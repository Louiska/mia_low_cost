from torchvision.models import resnet18, resnet50
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
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 44)
    elif architecture == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 44)
    else:
        print("Architecture not supported, try resnet18 or resnet50")
        exit(1)
    if pretrained_model != None:
        model.load_state_dict(torch.load(f"out/models/{pretrained_model}.pt", map_location="cpu"))
    
    return model
