from torchvision.models import resnet18
import torch

# load the data from a file
def get_model(name=""):
    if name != "":
        pretrained = False
    else:
        pretrained = True
    model = resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(512, 44)
    if name != "":
        model.load_state_dict(torch.load(f"out/models/{name}.pt", map_location = "cpu"))
    return model