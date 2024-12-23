from dataset import TaskDataset
from model import get_model
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from tqdm import tqdm


# ids, imgs, labels, membership
data: TaskDataset = torch.load("out/data/01/pub.pt")
ids = data.ids

# data: TaskDataset = torch.load("out/data/01/priv_out.pt")

if None in data.membership:
    print("***")
    print("Filling Nones with -1")  # Otherwise the dataloader is unhappy
    print("***")
    data.membership = [-1 if x is None else x for x in data.membership]

model = get_model()

dataloader = DataLoader(data, batch_size=64)
criterion = CrossEntropyLoss()
softmax = Softmax(dim=1)
batch_non_membership_loss, batch_membership_loss = 0, 0


def run_dataloader(dataloader, model):
    total_correct_mem_pred, total_correct_non_mem_pred, total_mem, total_non_mem = (
        0,
        0,
        0,
        0,
    )

    for _, imgs, labels, membership in tqdm(dataloader):
        outputs = model(imgs)
        logits = softmax(outputs)
        prediction = torch.argmax(outputs, dim=1)

        batch_indices = torch.arange(outputs.size(0))
        confidence = logits[batch_indices, prediction]

        correct_pred = prediction == labels

        if -1 not in membership:  # = evaluation mode
            membership = membership.bool()

            total_correct_mem_pred += sum(correct_pred[membership])
            total_mem += sum(membership)
            total_correct_non_mem_pred += sum(correct_pred[~membership])
            total_non_mem += sum(~membership)
    # Membership of training dataset seems to not have a, membership very high impact on the prediction results, in general the predictions seem very poor
    print(f"Correct prediction and member: {total_correct_mem_pred}/{total_mem}")
    print(
        f"Correct prediction and not member: {total_correct_non_mem_pred}/{total_non_mem}"
    )


run_dataloader(dataloader, model)
"""
precision = tp/(tp+fp)
recall = tp/(tp+fn)

print(f"TP:{tp}, FP:{fp}, FN:{fn}, Precision:{precision}, Recall{recall}")
"""
# print(f"Correct prediction and member: {sum(correct_pred[membership] == True)}/{sum(membership)}")
# print(f"Correct prediction and not member: {sum(correct_pred[~membership] == True)}/{sum(membership == 0)}")

# batch_membership_loss += outputs[membership]/sum(membership)
# print(batch_membership_loss)
# batch_non_membership_loss += outputs[~membership]/sum(~membership)

# avg_non_membership_loss = batch_non_membership_loss/idx
# avg_membership_loss = batch_membership_loss/idx

# print(avg_non_membership_loss)
# print(avg_membership_loss)
# loss = criterion(outputs, labels)
# loss.backward()
