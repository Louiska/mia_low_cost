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

if False:
    ids, labels = analyse_dataset(trainset)
    ids_mia, labels = analyse_dataset(targetset)
    concat_ids = ids + ids_mia
    print(
        f"Overlapping data according to id: {len(ids) + len(ids_mia) -len(set(concat_ids))}"
    )


def calc_pr_z(models, z, label: int) -> float:
    acc_sum = 0
    softmax = Softmax(dim=1)

    for model in models:
        output = model(z)
        acc_sum += softmax(output)[0, label]
    return acc_sum / len(models)


def calc_pr_x(models, x, label: int, alpha: float) -> float:
    pr_x_out = calc_pr_z(models, x, label)
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
    Z: int = 10000,
    alpha: float = 1,
):
    print("Starting MIA")
    ids = {}
    print(f"Z:{Z}, a:{alpha}")
    mia_subset, mia_subset_ids = create_subset(targetset, Z)
    z_ratios = {}
    print("Calculating z-Ratios")
    for id_z, z, z_label, _ in tqdm(mia_subset):
        z = z.unsqueeze(0)
        pr_z = calc_pr_x(
            shadow_models, z, z_label, alpha=alpha
        )  # correct to use z_out models?
        ratio_z = calc_ratio(target_model, z, z_label, pr_z)
        z_ratios[id_z] = ratio_z
    print("Done")
    for id_x, x, label, _ in tqdm(targetset):
        counter = 0
        x = x.unsqueeze(0)  # add batch layer
        pr_x = calc_pr_x(shadow_models, x, label, alpha=alpha)
        ratio_x = calc_ratio(target_model, x, label, pr_x)
        for key, ratio_z in z_ratios.items():
            if ratio_x / ratio_z > 1:
                counter += 1
        mia_score = counter / Z
        ids[id_x] = mia_score

    df = pd.DataFrame(ids.items(), columns=["ids", "score"])
    df.to_csv(f"{save_name}.csv", index=None)


def main(
    trainset_path: str,
    targetset_path: str,
    shadow_models_name: str,
    num_shadow_models: int,
    train_models: bool = False,
):
    random.seed(0)
    transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset: MembershipDataset = torch.load(trainset_path)
    targetset: MembershipDataset = torch.load(targetset_path)
    trainset.transform = transforms
    targetset.transform = transforms
    targetset.membership = [-1 if x is None else x for x in targetset.membership]
    shadow_models_save_path = f"out/models/{shadow_models_name}"
    trainset_size = 0.5
    if train_models:
        for i in range(num_shadow_models):
            indices_member, indices_non_member = [], []
            for k, is_member in enumerate(trainset.membership):
                if is_member:
                    indices_member.append(k)
                else:
                    indices_non_member.append(k)
            ids_member = random.sample(
                indices_member, int(len(indices_member) * trainset_size)
            )
            ids_non_member = random.sample(
                indices_non_member, int(len(indices_non_member) * trainset_size)
            )
            ids = ids_member + ids_non_member
            sub_trainset = Subset(trainset, ids)
            print(f"Training shadow model: {i}/{num_shadow_models}")
            train_shadow_model(
                get_model(""),
                sub_trainset,
                targetset,
                f"{shadow_models_save_path}_{i}",
                num_epochs=5,
                bs=64,
                lr=0.00004,
            )
    shadow_models = []
    for i in range(num_shadow_models):
        shadow_models.append(get_model(f"{shadow_models_name}_{i}").eval())
    target_model = get_model("target").eval()  # lol so many models in RAM
    mia(targetset, shadow_models, target_model, save_name=shadow_models_name)


trainset_path = "out/data/01/pub.pt"
targetset_path = "out/data/01/priv_out.pt"
main(
    trainset_path,
    targetset_path,
    shadow_models_name="model_v9",
    num_shadow_models=4,
    train_models=False,
)
