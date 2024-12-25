from torch.nn import Softmax, Module
from dataset_helper import create_subset
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd


def calc_pr_x_given_theta(models: list[Module], x, label: int) -> float:
    acc_sum = 0
    softmax = Softmax(dim=1)

    for model in models:
        output = model(x)
        acc_sum += softmax(output)[0, label]
    return acc_sum / len(models)


def calc_pr_x_offline(models: list[Module], x, label: int, alpha: float) -> float: 
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
    Z_set, _ = create_subset(Z_set, Z)
    z_ratios = {}
    print("Calculating z-Ratios")
    for id_z, z, z_label, _ in tqdm(Z_set):
        z = z.unsqueeze(0)
        if len(shadow_models) == 1:  # For single model, approximate pr_in
            pr_z = calc_pr_x_offline(shadow_models, z, z_label, alpha=alpha)
        else:
            pr_z = calc_pr_x_given_theta(shadow_models, z, z_label)
            # Equals pr_z as pr_z contains in and out models given that the shadow_models are in and out models
        ratio_z = calc_ratio(target_model, z, z_label, pr_z)
        z_ratios[id_z] = ratio_z
    print("Done")
    for id_x, x, label, _ in tqdm(targetset):
        counter = 0
        x = x.unsqueeze(0)
        pr_x = calc_pr_x_offline(shadow_models, x, label, alpha=alpha)
        ratio_x = calc_ratio(target_model, x, label, pr_x)
        for key, ratio_z in z_ratios.items():
            if ratio_x / ratio_z > 1:
                counter += 1
        mia_score = counter / Z
        ids[id_x] = mia_score

    df = pd.DataFrame(ids.items(), columns=["ids", "score"])
    df.to_csv(f"{save_name}.csv", index=None)
