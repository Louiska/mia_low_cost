from coding_task_1 import MembershipDataset
import torch
import matplotlib.pyplot as plt


def show_img(img):
    plt.imshow(img.permute(1,2,0))
    plt.show()