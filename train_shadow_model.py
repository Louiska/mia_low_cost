import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from torch import optim
from dataset_helper import create_subset
import csv
def train_shadow_model(model, trainset, valset, save_path = "", num_epochs = 20, bs = 64, lr = 0.004):
    print("Training model for: " + save_path)
    trainloader = DataLoader(trainset, batch_size = bs, num_workers = 16)
    valloader = DataLoader(valset, batch_size = bs, num_workers = 16)
    dataloader = {"train": trainloader, "val": valloader}
    criterion = CrossEntropyLoss()
    #softmax = Softmax(dim=1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = lr, 
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience = 3, factor = 0.4, verbose=True,
    )
    best_epoch_metric = 1000
    best_epoch = 0
    for epoch in range(num_epochs):
        print(f"Run epoch: {epoch}/{num_epochs}")
        for phase in ["train", "val"]:
            epoch_metric = run_epoch(model, 
                        phase,
                        dataloader[phase],
                        criterion,
                        optimizer, 
                        epoch)
            if phase == "train":
                train_epoch_metric = epoch_metric
            if phase == "val":
                scheduler.step(epoch_metric)
                if epoch_metric<best_epoch_metric:
                    best_epoch_metric = epoch_metric
                    best_epoch = epoch
                    counter_epoch_metric = 0
                    print(f"New best metric")
                    #torch.save(model.state_dict(), save_path + "_tmp.pt")
                    print("Tmp-Saved Model")
                else:
                    counter_epoch_metric+=1
                    print(f"No improvement for {counter_epoch_metric} epochs at lr {scheduler.get_last_lr()}")
                    
    torch.save(model.state_dict(), save_path + ".pt")
    print("Done, saved at " + save_path)
    data = {"name": save_path,
            "dataset_size":0.5, 
            "num_epochs": num_epochs,
            "lr": lr,
            "bs": bs,
            "last_train_metric": train_epoch_metric,
            "last val_metric": epoch_metric,
            "best_epoch_metric": best_epoch_metric,
            "best_epoch": best_epoch,
            "TPR@FPR=0.05":0,
            "AUC":0.5}
    with open("train_log.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    

def run_epoch(model, phase, dataloader, criterion, optimizer, epoch):
    total_correct, total_loss = 0,0
    for idx, (_, imgs, labels, _) in enumerate(tqdm(dataloader)):
        with torch.set_grad_enabled(
            phase == "train"
        ):  
            #with torch.autocast(device_type="cpu", dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if phase == "train":
                loss.backward()
                optimizer.step()      
                optimizer.zero_grad() 
            prediction = torch.argmax(outputs, dim = 1)
            total_correct += torch.sum(prediction == labels)
            total_loss += loss.item()
            if idx % 25 == 0:
                print(loss.item())
                print(f"Correct predictions: {total_correct}/{(idx+1)*imgs.size(0)}")
            
    print(f"Acc: {total_correct/(dataloader.batch_size*idx)}")
    print(f"Loss: {total_loss/idx}")
    return total_loss/idx
        

