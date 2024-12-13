from coding_task_1 import get_model,TaskDataset, MembershipDataset
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
    return softmax(output)[0, label]/pr_target

def mia(targetset, shadow_models, target_model, Z=10, alpha=0.3):    
    ids = {}
    for id_x, x, label,_ in tqdm(targetset):
        counter = 0
        x = x.unsqueeze(0) # add batch layer
        pr_x = calc_pr_x(shadow_models, x, label, alpha = alpha)
        ratio_x = calc_ratio(target_model, x, label, pr_x)
        mia_subset, mia_subset_ids = create_subset(targetset, Z)
        for id_z, z, z_label, _ in mia_subset: #randomly sampled & id_z != id_x
            z = z.unsqueeze(0) 
            #start = time.time()
            pr_z = calc_pr_z(shadow_models, z, z_label)
            #end = time.time()
            #print(end-start)
            #start = time.time()
            ratio_z = calc_ratio(target_model, z, z_label, pr_z)
            #end = time.time()
            #print(end-start)
            #print(f"pr_z: {pr_z}, ratio_z: {ratio_z}")
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
    trainset:MembershipDataset = torch.load(trainset_path)
    targetset:MembershipDataset = torch.load(targetset_path)
    targetset.membership = [-1 if x is None else x for x in targetset.membership]
    shadow_models = [get_model("shadow").eval(), get_model("shadow2").eval()]
    target_model = get_model("target").eval() # lol 3 models in RAM
    mia(targetset, shadow_models, target_model)

trainset_path = "out/data/01/pub.pt"
targetset_path = "out/data/01/priv_out.pt"
main(trainset_path, targetset_path, "target", ["shadow", "shadow2"])