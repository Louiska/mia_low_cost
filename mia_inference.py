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

def mia(targetset, shadow_models, target_model, Z=10, alpha=0.3, member_subset= []):    
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

def main(trainset_path, targetset_path, targetmodel_name, shadowmodels_name = []):
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
    indices = [i for i, x in enumerate(trainset.membership) if x == 1]
    trainset = Subset(trainset, indices) #TODO ensure that 50% is member, 50% is not member
    print(len(trainset))
    #subset1, ids1= create_subset(trainset, int(len(trainset)/2))
    #subset2, ids2= create_subset(trainset, int(len(trainset)/2))
    #train_shadow_model(get_model("target"), targetset, trainset, "out/models/tmp.pt")
    #train_shadow_model(get_model(""), trainset, targetset, "out/models/shadow_full_v3")
    #shadow_models = [get_model("shadow").eval(), get_model("shadow2").eval()]
    shadow_models = [get_model("shadow_full_v3_best_epochmetric").eval()]
    target_model = get_model("target").eval() # lol 3 models in RAM
    mia(targetset, shadow_models, target_model, member_subset=trainset)

trainset_path = "out/data/01/pub.pt"
targetset_path = "out/data/01/priv_out.pt"
main(trainset_path, targetset_path, "target", ["shadow", "shadow2"])