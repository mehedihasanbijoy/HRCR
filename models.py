import torch
import torch.nn as nn
import warnings as wrn
wrn.filterwarnings('ignore')


class HRCRBasic(nn.Module):
    def __init__(self, classes=50):
        super(HRCRBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.1)

    def forward(self, img):  # img: [40 x 40 x 3]
        img = self.maxpool(self.relu(self.dropout(self.conv1(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv2(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv3(img))))
        img = self.relu(self.dropout(self.conv4(img)))
        img = img.reshape(img.shape[0], -1)
        img = self.relu(self.dropout(self.fc1(img)))
        img = self.fc2(img)
        return img


if __name__ == '__main__':
    random_batch = torch.randn(32, 3, 40, 40)
    model = HRCRBasic(classes=20)
    model_output = model(random_batch)
    print(model_output.shape)
