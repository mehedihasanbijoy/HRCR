from tqdm import tqdm
import torch


def train(model, optimizer, criterion, dataloader, device):
    model.train()
    epoch_loss = 0
    train_count, correct_preds = 0, 0

    for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device)
        _, labels = torch.max(labels.data, 1)
        labels = labels.to(device)

        outputs = model(imgs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * labels.shape[0]

        _, preds = torch.max(outputs.data, 1)
        train_count += labels.shape[0]
        correct_preds += (preds == labels).sum().item()

    epoch_acc = correct_preds / train_count
    epoch_loss = epoch_loss / train_count
    return epoch_loss, epoch_acc
