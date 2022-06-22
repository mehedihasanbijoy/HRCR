from tqdm import tqdm
import torch


def evaluate(model, dataloader, device):
    model.eval()
    eval_count, correct_preds = 0, 0

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            _, preds = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)

            eval_count += labels.shape[0]
            correct_preds += (preds == labels).sum().item()

    epoch_acc = correct_preds / eval_count
    return epoch_acc

