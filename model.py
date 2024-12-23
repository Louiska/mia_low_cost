from torchvision.models import resnet18
import torch
from torch.nn import Module

# load the data from a file
def get_model(name: str = "") -> Module:
    """
    Args:
        name (str, optional): File name of model that's supposed to be loaded. If "", 
        only loads pretrained weights. Defaults to "".

    Returns:
        (Module): The loaded model
    """
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 44)
    if name != "":
        model.load_state_dict(torch.load(f"out/models/{name}.pt", map_location="cpu"))
    return model
