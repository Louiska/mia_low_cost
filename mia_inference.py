from dataset import TaskDataset, MembershipDataset
from model import get_model
from dataset_helper import create_subset
from train_shadow_model import train_shadow_model
from visualize import show_img
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss, Softmax
import torchvision
import numpy as np
import pandas as pd
import cv2
import random
from tqdm import tqdm
import time
import pickle
from torchvision.transforms import v2
if False:
    ids, labels = analyse_dataset(trainset)
    ids_mia, labels = analyse_dataset(targetset)
    concat_ids = ids + ids_mia
    print(f"Overlapping data according to id: {len(ids) + len(ids_mia) -len(set(concat_ids))}")


def calc_pr_z(models, z, label):
    acc_sum = 0
    softmax = Softmax(dim=1)
        
    for model in models:
        output = model(z)
        acc_sum += softmax(output)[0, label]
    return acc_sum/len(models)

def calc_pr_x(models, x, label, alpha):
    pr_x_out = calc_pr_z(models, x, label)
    return 0.5*((1+alpha)*pr_x_out+(1-alpha))

def calc_ratio(target_model, target, label, pr_target):
    softmax = Softmax(dim=1)
    output = target_model(target)
    return (softmax(output)[0, label]/pr_target).item()

def mia(targetset, shadow_models, target_model, Z=1000, alpha=0.3):    
    print("Starting MIA")
    ids = {}
    print(f"Z:{Z}, a:{alpha}")
    mia_subset, mia_subset_ids = create_subset(targetset, Z)
    z_ratios = {}
    print("Calculating z-Ratios")
    for id_z, z, z_label, _ in tqdm(mia_subset):
        z = z.unsqueeze(0) 
        pr_z = calc_pr_z(shadow_models, z, z_label)
        ratio_z = calc_ratio(target_model, z, z_label, pr_z)
        z_ratios[id_z] = ratio_z
    print("Done")
    for id_x, x, label,_ in tqdm(targetset):
        counter = 0
        x = x.unsqueeze(0) # add batch layer
        pr_x = calc_pr_x(shadow_models, x, label, alpha = alpha)
        ratio_x = calc_ratio(target_model, x, label, pr_x)
        if ratio_x/ratio_z>1:
            counter+=1
        mia_score = counter/Z
        ids[id_x] = mia_score
    with open("member_pred.pkl","wb") as f:
        pickle.dump(ids, f)
    print(ids)

    df = pd.DataFrame(ids.items(), columns = ["ids", "score"])
    df.to_csv("test.csv", index=None)
#TODO interdependenz z & x samples? Hängen sie irgendwie voneinander ab?
#TODO Ist das "is member" Attribut irgendwie relevant?
#trainsets, overlap und co
#TODO Z verdoppeln, alpha anpassen?
#TODO Test how well it would perform on the training set (TPR@FPR=0.05)
#Usually Z 5000, alpha 0.3
#TODO show accuracy and loss of members/non members
#TODO Log configs and responses
#TODO train multiple models
def main(trainset_path, targetset_path, targetmodel_name, shadow_models_name = [], train_models = False):
    random.seed(0)
    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset:MembershipDataset = torch.load(trainset_path)
    targetset:MembershipDataset = torch.load(targetset_path)
    trainset.transform = transforms
    targetset.transform = transforms
    targetset.membership = [-1 if x is None else x for x in targetset.membership]
    
    if train_models:
        for name in shadow_models_name:
            train_shadow_model(get_model(""),trainset, targetset, name, num_epochs= 20, bs= 64, lr = 0.004)
    shadow_models = []
    for name in shadow_models_name: 
        shadow_models.append(get_model(name).eval())
    target_model = get_model("target").eval() # lol so many models in RAM
    mia(targetset, shadow_models, target_model)


trainset_path = "out/data/01/pub.pt"
targetset_path = "out/data/01/priv_out.pt"
main(trainset_path, targetset_path, "target", ["shadow", "shadow2"], train_models = True)