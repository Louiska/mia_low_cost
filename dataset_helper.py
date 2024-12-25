from torch.utils.data import Subset, Dataset
import random
import torch
from dataset import MembershipDataset


def create_subset(trainset: Dataset, k: int):
    selected_ids = random.sample(range(len(trainset)), k)
    subset = Subset(trainset, selected_ids)
    return subset, selected_ids


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
        print(
            f"Splitting dataset in {k} of proportional {trainset_size} size. With replacement!"
        )
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
