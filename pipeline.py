from train import train
from evaluate import evaluate
from dataloaders import CustomDataset
from models import HRCRBasic
import pandas as pd
import torch
import torchvision


def basic_classification(transform, n_classes, epochs, batch_size, device):
    df_train = pd.read_csv('./dfs/train.csv')
    df_test = pd.read_csv('./dfs/test.csv')

    dataset_train = CustomDataset(df=df_train, transform=transform)
    dataset_test = CustomDataset(df=df_test, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size,
        shuffle=True, drop_last=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size,
        shuffle=True, drop_last=False, pin_memory=True
    )

    model = HRCRBasic(classes=n_classes).to(device)  # basic model

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs+1):
        print(f"Epoch: {epoch} / {epochs}")
        train_loss, train_acc = train(
            model, optimizer, criterion, train_loader, device
        )
        print(f"Training Loss: {train_loss:.2f}, Training Accuracy: {train_acc*100:.2f}%")

        if epoch%5 == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Test Accuracy: {test_acc*100:.2f}%")