from torch.utils.data import Subset, Dataset
import random

def create_subset(trainset: Dataset, k: int):
    selected_ids = random.sample(range(len(trainset)), k)
    subset = Subset(trainset, selected_ids)
    return subset, selected_ids
