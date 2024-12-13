
from torch.utils.data import Subset
import random

def analyse_dataset(dataset):
    labels = {i:dataset.labels.count(i) for i in dataset.labels}
    print(f"Length: {len(dataset)}")
    print("Label: n")
    print(labels)
    return dataset.ids, labels


def create_subset(trainset, k):
    selected_ids = random.sample(range(len(trainset)), k)
    subset = Subset(trainset, selected_ids)
    return subset, selected_ids