
import torch
import matplotlib.pyplot as plt

means = torch.tensor([0.485, 0.456, 0.406])
stds = torch.tensor([0.229, 0.224, 0.225])

def show_img(img:torch.tensor):
    img = img.permute(1,2,0)
    img = img * stds + means
    print(img)
    plt.imshow(img)
    plt.show()