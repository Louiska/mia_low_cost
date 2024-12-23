from torch.utils.data import Subset, Dataset
import random


def analyse_dataset(dataset: Dataset):
    labels = {i: dataset.labels.count(i) for i in dataset.labels}
    print(f"Length: {len(dataset)}")
    print("Label: n")
    print(labels)
    return dataset.ids, labels


def create_subset(trainset: Dataset, k: int):
    selected_ids = random.sample(range(len(trainset)), k)
    subset = Subset(trainset, selected_ids)
    return subset, selected_ids
