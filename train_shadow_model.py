import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW, lr_scheduler
import csv


def train_shadow_model(
    model: Module,
    trainset: Dataset,
    valset: Dataset,
    save_path: str,
    num_epochs: int,
    bs: int,
    lr: float,
):
    """Trains a model, logs its hyperparameters and metrics in a csv called train_log.csv

    Args:
        model (Module): Model to be trained
        trainset (Dataset): Trainset
        valset (Dataset): Validationset
        save_path (str): Path for saving model.
        num_epochs (int): Total number of epochs to train.
        bs (int): Batch size.
        lr (float): Learning rate.
    """
    print("Training model for: " + save_path)
    trainloader = DataLoader(trainset, batch_size=bs, num_workers=16)
    valloader = DataLoader(valset, batch_size=bs, num_workers=16)
    dataloader = {"train": trainloader, "val": valloader}
    criterion = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
    )

    th = 0.02
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,  # if no improvement for patience epochs
        factor=0.5,  # factor of how much its reduced
        threshold=th,  # the relative change to the ath
    )
    best_epoch_metric = 0
    lowest_loss = 1000
    best_epoch = 0
    for epoch in range(num_epochs):
        print(f"Run epoch: {epoch}/{num_epochs}")
        for phase in ["train", "val"]:
            loss, epoch_metric = run_epoch(
                model, phase, dataloader[phase], criterion, optimizer, epoch
            )
            if phase == "train":
                train_epoch_metric = epoch_metric
                train_loss = loss
            if phase == "val":
                scheduler.step(epoch_metric)
                if loss < lowest_loss * (1 - th):  # threshold of scheduler
                    best_epoch_metric = (
                        epoch_metric  # FIXME best epoch doesnt have to be current epoch
                    )
                    best_epoch = epoch
                    lowest_loss = loss
                    counter_epoch_metric = 0
                    print(f"New best metric")
                    torch.save(model.state_dict(), save_path + "_best.pt")
                    print("Tmp-Saved Model")
                else:
                    counter_epoch_metric += 1
                    print(
                        f"No {th} improvement relative to {lowest_loss} for {counter_epoch_metric} epochs at lr {scheduler.get_last_lr()}"
                    )

    torch.save(model.state_dict(), save_path + ".pt")
    print("Done, saved at " + save_path)
    data = {
        "name": save_path,
        "dataset_size": 1,
        "num_epochs": num_epochs,
        "lr": lr,
        "bs": bs,
        "last_train_metric": train_epoch_metric,
        "last_train_loss": train_loss,
        "last val_metric": epoch_metric,
        "last_val_loss": loss,
        "best_epoch_metric": best_epoch_metric,
        "lowest_loss": lowest_loss,
        "best_epoch": best_epoch,
        "TPR@FPR=0.05": 0,
        "AUC": 0.5,
    }
    log_data(data)


def log_data(data: dict):
    """Saves the given data to train_log.csv

    Args:
        data (dict): Contains structured data
    """
    with open("train_log.csv", "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data)


def run_epoch(
    model: Module, phase: str, dataloader, criterion, optimizer, epoch: int
) -> tuple[float, float]:
    """Runs an epoch, either for training or validation

    Args:
        model (Module): Model for training
        phase (str): Train or validation
        dataloader (_type_): Dataloader_
        criterion (_type_): Loss function
        optimizer (_type_): Optimizer
        epoch (int): Current epoch

    Returns:
        tuple[float, float]: Average loss per sample, overall accuracy for epoch
    """
    total_correct, total_loss = 0, 0
    for idx, (_, imgs, labels, _) in enumerate(tqdm(dataloader)):
        with torch.set_grad_enabled(phase == "train"):
            # with torch.autocast(device_type="cpu", dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if phase == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prediction = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(prediction == labels)
            total_loss += loss.item()
            if idx % 25 == 0:
                print(loss.item())
                print(f"Correct predictions: {total_correct}/{(idx+1)*imgs.size(0)}")
    acc = total_correct / (dataloader.batch_size * idx)
    print(f"Acc: {acc}")
    print(f"Loss: {total_loss/idx}")
    return total_loss / idx, acc
