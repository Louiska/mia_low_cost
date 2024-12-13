import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from torch import optim
from dataset_helper import create_subset



def train_shadow_model(model, trainset, valset, num_epochs = 10): 
    trainloader = DataLoader(trainset, batch_size = 64, num_workers = 16)
    valloader = DataLoader(valset, batch_size = 64, num_workers = 16)
    dataloader = {"train": trainloader, "val": valloader}
    criterion = CrossEntropyLoss()
    softmax = Softmax(dim=1)
    optimizer = optim.SGD(
        model.parameters(),
        lr = 0.001,
        weight_decay = 0.4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience = 3, factor = 0.2
    )
    for epoch in range(num_epochs):
        print(f"Run epoch: {epoch}/{num_epochs}")
        for phase in ["train", "val"]:
            run_epoch(model, 
                        phase,
                        dataloader[phase],
                        criterion, 
                        scheduler,
                        optimizer, 
                        epoch)

def run_epoch(model, phase, dataloader, criterion, scheduler, optimizer, epoch):
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
                print(loss)
                print(f"Correct predictions: {total_correct}/{(idx+1)*imgs.size(0)}")
            
    print(f"Acc: {total_correct/(dataloader.batch_size*idx)}")
    print(f"Loss: {total_loss/idx}")
    scheduler.step(total_loss/idx)
        


