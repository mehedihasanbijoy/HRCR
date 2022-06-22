import torch
import torchvision
import cv2
import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = pd.concat([df, pd.get_dummies(df.iloc[:, 1])], axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = cv2.imread(img_path)
        img = self.transform(img)
        label = torch.from_numpy(self.df.iloc[idx, 2:].values.astype(float))
        return img, label


if __name__ == '__main__':
    df = pd.read_csv('./dfs/train.csv')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor()
    ])
    dataset = CustomDataset(df=df, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32,
        shuffle=True, drop_last=True, pin_memory=True
    )
    for imgs, labels in train_loader:
        print(imgs.shape)
        print(labels.shape)
        break