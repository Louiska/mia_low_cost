import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from torch import optim
from dataset_helper import create_subset

def train_shadow_model(model, trainset, valset, save_path = "", num_epochs = 30):
    print("Training model for: " + save_path)
    trainloader = DataLoader(trainset, batch_size = 64, num_workers = 16)
    valloader = DataLoader(valset, batch_size = 64, num_workers = 16)
    dataloader = {"train": trainloader, "val": valloader}
    criterion = CrossEntropyLoss()
    #softmax = Softmax(dim=1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = 0.004, 
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience = 3, factor = 0.4, verbose=True,
    )
    hs_epoch_metric = 1000
    for epoch in range(num_epochs):
        print(f"Run epoch: {epoch}/{num_epochs}")
        for phase in ["train", "val"]:
            epoch_metric = run_epoch(model, 
                        phase,
                        dataloader[phase],
                        criterion,
                        optimizer, 
                        epoch)
            if phase == "val":
                scheduler.step(epoch_metric)
                if epoch_metric<hs_epoch_metric:
                    hs_epoch_metric = epoch_metric
                    counter_epoch_metric = 0
                    print(f"New best metric")
                    #torch.save(model.state_dict(), save_path + "_tmp.pt")
                    print("Tmp-Saved Model")
                else:
                    counter_epoch_metric+=1
                    print(f"No improvement for {counter_epoch_metric} epochs at lr {scheduler.get_last_lr()}")

                #if epoch+1 % 6 == 0:
                #    torch.save(model.state_dict(), save_path + "_tmp_" +str(epoch) + ".pt")
                    
    torch.save(model.state_dict(), save_path + ".pt")
    print("Done, saved at " + save_path)

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
        

