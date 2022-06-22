from pipeline import basic_classification
import torch
import torchvision


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 32

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((40, 40)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor()
    ])

    basic_classification(
        transform=transform,
        n_classes=50,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )