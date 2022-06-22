from pipeline import classification
from models import HRCRBasic
import torch
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 32

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((40, 40)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor()
])


def basic_classifier():
    model = HRCRBasic(classes=50)
    classification(
        model=model, transform=transform, batch_size=batch_size,
        epochs=epochs, device=device
    )


if __name__ == '__main__':
    basic_classifier()