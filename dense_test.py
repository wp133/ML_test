import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_densenet(num_classes=10):
    model = models.densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def train(model, dataloader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100  # CIFAR-100 has 100 fine-grained classes

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  #resize 32x32 -> 64x64 for DenseNet compatibility
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #train_path = "hf://datasets/uoft-cs/cifar100/cifar100/train-00000-of-00001.parquet"
    #test_path = "hf://datasets/uoft-cs/cifar100/cifar100/test-00000-of-00001.parquet"

    train_path = "train-00000-of-00001.parquet"
    test_path = "test-00000-of-00001.parquet"

    def load_parquet(path):
        df = pd.read_parquet(path)
        from PIL import Image
        import io
        
        images = []
        for img_dict in df["img"]:
            img_bytes = img_dict["bytes"]
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)  # Shape: (32, 32, 3), dtype: uint8
            images.append(img_array)
        
        images = np.stack(images)  # Shape: (N, 32, 32, 3)
        fine_labels = df["fine_label"].to_numpy()
        coarse_labels = df["coarse_label"].to_numpy()
        
        #convert HWC to CHW 
        images = images.transpose(0, 3, 1, 2)  # Shape: (N, 3, 32, 32)
        
        images = torch.from_numpy(images).float() / 255.0
        fine_labels = torch.from_numpy(fine_labels).long()
        coarse_labels = torch.from_numpy(coarse_labels).long()
        
        return images, fine_labels, coarse_labels

    train_images, train_fine_labels, train_coarse_labels = load_parquet(train_path)
    test_images, test_fine_labels, test_coarse_labels = load_parquet(test_path)

    train_dataset = ImageDataset(train_images, train_fine_labels, transform=transform)
    test_dataset = ImageDataset(test_images, test_fine_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = build_densenet(num_classes=num_classes).to(device)
    train(model, train_loader, device, epochs=10)
