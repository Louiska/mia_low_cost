# Trains and executes MIA on a given training-/targetset

from dataset import MembershipDataset
from model import get_model
from dataset_helper import create_subset
from train_shadow_model import train_shadow_model
import torch
from torch.utils.data import Subset, Dataset
from torch.nn import Softmax, Module
import pandas as pd
import random
from tqdm import tqdm
from torchvision.transforms import v2
from visualize import show_img


def calc_pr_x_given_theta(models: list[Module], x, label: int) -> float:
    acc_sum = 0
    softmax = Softmax(dim=1)

    for model in models:
        output = model(x)
        acc_sum += softmax(output)[0, label]
    return acc_sum / len(models)


def calc_pr_out(models: list[Module], x, label: int, alpha: float) -> float:
    pr_x_out = calc_pr_x_given_theta(models, x, label)
    return 0.5 * ((1 + alpha) * pr_x_out + (1 - alpha))


def calc_ratio(target_model, target, label: int, pr_target: float) -> float:
    softmax = Softmax(dim=1)
    output = target_model(target)
    return (softmax(output)[0, label] / pr_target).item()


def mia(
    targetset: Dataset,
    shadow_models: list[Module],
    target_model: Module,
    save_name: str,
    Z_set: Dataset,
    Z: int,
    alpha: float,
):
    """
    Calculate MIA in offline mode as shown by Zarifzadeh et al. in https://doi.org/10.48550/arXiv.2312.03262

    Args:
        targetset (Dataset): Contains target samples to be classified if part of the trainingset of the target model or not
        shadow_models (list[Module]): Aka. reference models, trained with a dataset of (ideally) the same distribution as the target model
        target_model (Module): The model we try to extract the training data from
        save_name (str): Where to save the resulting csv
        Z_set (Dataset): Optionally a subset of the trainset as Z set
        Z (int, optional): How many reference samples will be used.
        alpha (float, optional): To approximate the effect of the inner model.
    """
    print("Starting MIA")
    ids = {}
    print(f"Z_set: {len(Z_set)}, Z:{Z}, a:{alpha}")
    Z_set, mia_subset_ids = create_subset(Z_set, Z)
    z_ratios = {}
    print("Calculating z-Ratios")
    for id_z, z, z_label, _ in tqdm(Z_set):
        # show_img(z)
        z = z.unsqueeze(0)
        pr_z = calc_pr_x_given_theta(shadow_models, z, z_label)
        ratio_z = calc_ratio(target_model, z, z_label, pr_z)
        z_ratios[id_z] = ratio_z
    print("Done")
    for id_x, x, label, _ in tqdm(targetset):
        counter = 0
        x = x.unsqueeze(0)
        pr_x = calc_pr_out(shadow_models, x, label, alpha=alpha)
        ratio_x = calc_ratio(target_model, x, label, pr_x)
        for key, ratio_z in z_ratios.items():
            if ratio_x / ratio_z > 1:
                counter += 1
        mia_score = counter / Z
        ids[id_x] = mia_score

    df = pd.DataFrame(ids.items(), columns=["ids", "score"])
    df.to_csv(f"{save_name}.csv", index=None)


def split_dataset(
    trainset: Dataset, replacement: bool = False, k=2, trainset_size=1
) -> list[Dataset]:
    """Split a dataset in k pieces, with or without replacement. 
    It always splits by keeping the (non)/meber ratio 50/50.

    Args:
        trainset (Dataset): Dataset to split.
        replacement (bool, optional): If replacement, the subsets (might) overlap. Defaults to False.
        k (int, optional): The amount of subsets created. Defaults to 2.
        trainset_size (int, optional): If replacement, how much of the dataset should be sampled for each subset. Defaults to 1.

    Returns:
        list[Dataset]: Returns k Subsets.
    """

    member_indices = [i for i, x in enumerate(trainset.membership) if x == 1]
    non_member_indices = [i for i, x in enumerate(trainset.membership) if x == 0]
    shadow_models_sets = []
    if replacement:
        print(f"Splitting dataset in {k} of proportional {trainset_size} size. With replacement!")
        for _ in range(k):  # if multiple shadowmodels are trained
            ids_member = random.sample(
                member_indices, int(len(member_indices) * trainset_size)
            )
            ids_non_member = random.sample(
                non_member_indices, int(len(non_member_indices) * trainset_size)
            )
            ids = ids_member + ids_non_member
            shadow_models_sets += Subset(trainset, ids)

    else:  # TODO allow more than 2 without replacement
        print("Currently only supports split by 2 without replacement")
        h_size = int(len(member_indices) / 2)
        shadow_1_indices = member_indices[:h_size] + non_member_indices[:h_size]
        shadow_2_indices = member_indices[h_size:] + non_member_indices[h_size:]

        shadow_1_set = Subset(trainset, shadow_1_indices)
        shadow_2_set = Subset(trainset, shadow_2_indices)
        shadow_models_sets = [shadow_1_set, shadow_2_set]

    return shadow_models_sets


def get_dataset(trainset_path, targetset_path, train_transforms, target_transforms):
    trainset: MembershipDataset = torch.load(trainset_path)
    targetset: MembershipDataset = torch.load(targetset_path)
    trainset.transform = train_transforms
    targetset.transform = target_transforms
    targetset.membership = [-1 if x is None else x for x in targetset.membership]
    return trainset, targetset


def get_transforms():
    train_transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomPerspective(),  # warps, cuts samples
            v2.AugMix(),  # tilts, colors, blurs samples
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard normalization for ResNet
        ]
    )
    target_transforms = v2.Compose(
        [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    return train_transforms, target_transforms


def main(
    trainset_path: str,
    targetset_path: str,
    shadow_models_name: str,
    num_shadow_models: int,
    train_models: bool = False,
):
    """Trains shadow models and forwards them to the MIA

    Args:
        trainset_path (str): Path to training set
        targetset_path (str): Path to set of to classify data
        shadow_models_name (str): Name for saving/loading shadow models
        num_shadow_models (int): Aka. k, how many reference models should be trained
        train_models (bool, optional): If true, trains shadow models, otherwise loads existing models. Defaults to False.
    """
    random.seed(0)
    train_transforms, target_transforms = get_transforms()
    trainset, targetset = get_dataset(
        trainset_path, targetset_path, train_transforms, target_transforms
    )

    if train_models:
        shadow_models_save_path = f"out/models/{shadow_models_name}"
        if num_shadow_models > 1:
            shadow_models_sets = split_dataset(
            trainset, replacement=True, k=num_shadow_models, trainset_size=1
        )  # Create trainset of trainset_size that contains the same ratio of member/non members
        else: 
            shadow_models_sets = trainset
        print(f"Training shadow model: {i}/{num_shadow_models}")
        for i, sub_trainset in enumerate(shadow_models_sets):
            train_shadow_model(
                get_model(""),
                sub_trainset,
                targetset,
                f"{shadow_models_save_path}_{i}",
                num_epochs=20,
                bs=64,
                lr=0.0001,
            )

    #Load shadow and target models
    shadow_models = []
    for i in range(num_shadow_models):
        shadow_models.append(get_model(f"{shadow_models_name}_{i}_best").eval())
    target_model = get_model("target").eval()  # lol so many models in RAM

    # Prepare Z set
    non_member_indices = [i for i, x in enumerate(trainset.membership) if x == 0]
    trainset.transform = target_transforms
    Z_set = Subset(trainset, non_member_indices) # No z in Z is part of the targets training data

    mia(
        targetset,
        shadow_models,
        target_model,
        Z_set=Z_set,
        save_name=shadow_models_name,
        Z=2000,
        alpha=0.3,
    )


trainset_path = "out/data/01/pub.pt"
targetset_path = "out/data/01/priv_out.pt"
shadow_model_name = "model_v13"

main(
    trainset_path,
    targetset_path,
    shadow_model_name,
    num_shadow_models=2,
    train_models=True,
)
