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


def build_simple_cnn(num_classes=100):
    """Alternative: Simple CNN optimized for CIFAR-100 (32x32 images)"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            )
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return SimpleCNN(num_classes)


def build_densenet(num_classes=100):
    """Custom DenseNet implementation faithful to the original algorithm"""
    class DenseBlock(nn.Module):
        def __init__(self, in_channels, growth_rate, num_layers):
            super(DenseBlock, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                # DenseNet-B: bottleneck (1×1 conv) + 3×3 conv
                self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, 4 * growth_rate, kernel_size=1, bias=False),  # bottleneck
                    nn.BatchNorm2d(4 * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
                ))
        
        def forward(self, x):
            features = [x]
            for layer in self.layers:
                new_feature = layer(torch.cat(features, 1))
                features.append(new_feature)
            return torch.cat(features, 1)
    
    class TransitionLayer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(TransitionLayer, self).__init__()
            self.layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.AvgPool2d(2),
            )
        
        def forward(self, x):
            return self.layer(x)
    
    class CustomDenseNet(nn.Module):
        def __init__(self, num_classes=100, growth_rate=32, block_config=(6, 12, 24, 16)):
            super(CustomDenseNet, self).__init__()
            
            # Initial convolution (same as original)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Dense blocks and transition layers
            num_features = 64
            self.dense_blocks = nn.ModuleList()
            self.transitions = nn.ModuleList()
            
            for i, num_layers in enumerate(block_config):
                self.dense_blocks.append(DenseBlock(num_features, growth_rate, num_layers))
                num_features += num_layers * growth_rate
                
                if i != len(block_config) - 1:
                    out_features = int(num_features * 0.5)  # DenseNet-C: compression factor 0.5
                    self.transitions.append(TransitionLayer(num_features, out_features))
                    num_features = out_features
            
            # Final layers
            self.bn_final = nn.BatchNorm2d(num_features)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(num_features, num_classes)
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            for dense_block, transition in zip(self.dense_blocks[:-1], self.transitions):
                x = dense_block(x)
                x = transition(x)
            
            x = self.dense_blocks[-1](x)
            x = self.bn_final(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return CustomDenseNet(num_classes=num_classes)


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
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    num_classes = 100  # CIFAR-100 has 100 fine-grained classes

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize 32x32 to 64x64 for DenseNet compatibility
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
        
        # Convert images from HWC to CHW format
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

    model = build_densenet(num_classes=num_classes).to(device)  # Now uses your custom DenseNet
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train(model, train_loader, device, epochs=10)

