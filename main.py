""" Used to train shadow models and call MIA on a given training-/targetset. """

from model import get_model
from train_shadow_model import train_shadow_model
from torch.utils.data import Subset
import random
from torchvision.transforms import v2
from mia import mia
from dataset_helper import split_dataset, get_dataset
from dataset import MembershipDataset

TRAINSET_PATH = "out/data/01/pub.pt"
TARGETSET_PATH = "out/data/01/priv_out.pt"
SHADOW_MODEL_NAME = "model_v15"
TARGET_ARCHITECTURE = "resnet18"
TARGET_MODEL_NAME = "target"


def main(
    trainset_path: str,
    targetset_path: str,
    shadow_models_save_name: str,
    num_shadow_models: int,
    model_architecture: str = "resnet18",
    train: bool = False
):
    """Trains shadow models and forwards them to the MIA

    Args:
        trainset_path (str): Path to training set
        targetset_path (str): Path to set of to classify data
        shadow_models_save_name (str): Name for saving/loading shadow models.
        num_shadow_models (int): Aka. k, how many reference models should be trained
        model_architecture (str, optional): The architecture of the model. Either resnet18 or resnet50. Defaults to resnet18
        train (bool, optional): If true, trains new shadow models. Defaults to False
    """
    random.seed(0)
    train_transforms, target_transforms = get_transforms()
    trainset, targetset = get_dataset(
        trainset_path, targetset_path, train_transforms, target_transforms
    )

    if train:
        shadow_models_save_path = f"out/models/{shadow_models_save_name}"
        if num_shadow_models > 1:
            shadow_models_sets = split_dataset(
                trainset, replacement=True, k=num_shadow_models, trainset_size=1
            )  # Create trainset of trainset_size that contains the same ratio of member/non members
        else:
            shadow_models_sets = [trainset]
        for i, shadow_model_set in enumerate(shadow_models_sets):
            print(f"Training shadow model: {i+1}/{num_shadow_models}")
            train_shadow_model(
                get_model(model_architecture),
                shadow_model_set,
                targetset,
                f"{shadow_models_save_path}_{i}",
                num_epochs=20,
                bs=64,
                lr=0.0001,
            )

    # Load shadow and target models
    shadow_models = []
    for i in range(num_shadow_models):
        shadow_models.append(
            get_model(model_architecture, f"{shadow_models_save_name}_{i}_best").eval()
        )
    target_model = get_model(
        TARGET_ARCHITECTURE, TARGET_MODEL_NAME
    ).eval()  # lol so many models in RAM

    # Prepare Z set
    non_member_indices = [i for i, x in enumerate(trainset.membership) if x == 0]
    trainset.transform = target_transforms
    Z_set = Subset(
        trainset, non_member_indices
    )  # No z in Z is part of the targets training data

    mia(
        targetset,
        shadow_models,
        target_model,
        Z_set=Z_set,
        save_name=shadow_models_save_name,
        Z=2000,
        alpha=0.8,
    )


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


main(
    TRAINSET_PATH,
    TARGETSET_PATH,
    SHADOW_MODEL_NAME,
    num_shadow_models=1,
    model_architecture="resnet50",
    train=True
)
