import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from torch.utils.data import Subset
from typing import Tuple, Dict, List

def main():
    #Naudojam GPU, jei nera CPU tada
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset dir path
    data_path = Path("raw-img/")

    # Pasidarom funkcija kuri printints dir ir failu skaiciu
    def dir_walk(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"Turime {len(dirnames)} direktorijas ir {len(filenames)} failus direktorijoje: {dirpath}")

    # dir_walk(data_path)

    #Pasidarom lista visu img paths
    image_path_list = list(data_path.glob("*/*.*"))

    # Pasirenkam random img
    random_image_path = random.choice(image_path_list)
    print(f"Random image path: {random_image_path}")

    # Pasiziurim img class (directory name sitam atveji)
    image_class = random_image_path.parent.stem
    print(f"Image class: {image_class}")

    # Open image
    img = Image.open(random_image_path)

    # Print img data
    print(f"Random image path: {random_image_path}")
    print(f"Random image class: {image_class}")
    print(f"Random image height: {img.height}")
    print(f"Random image width: {img.width}")

    # Apsirasom train ir test transformation
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
    ])

    def plot_transformed_images(image_paths, transform, n=3):
        random_image_paths = random.sample(image_paths, k=n)
        for image_path in random_image_paths:
            with Image.open(image_path) as img:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].imshow(img)
                ax[0].set_title(f"Original size:\n {img.size}")
                ax[0].axis(False)
                transformed_image = transform(img).permute(1, 2, 0)
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"Transformed Shape: \n {transformed_image.shape}")
                ax[1].axis(False)
                fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

    # Plot transformed images
    plot_transformed_images(image_paths=image_path_list, transform=data_transform, n=3)

    # Create dataset
    dataset = datasets.ImageFolder(root=data_path, transform=data_transform)

    # Get train and test indices
    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    # Pasidarom train ir test subsetus
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    class CustomImageDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            return image, label

    # Create dataset instances
    train_dataset = CustomImageDataset(dataset=train_dataset)
    test_dataset = CustomImageDataset(dataset=test_dataset)

    #Pasiziurim shapes
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    #Pasikuriam dataloaderius
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(train_dataloader)

    img_custom, label_custom = next(iter(train_dataloader))
    print(f"Batch shape: {img_custom.shape}")
    print(f"Label shape: {label_custom.shape}")

    #Aprasom modelio architektura
    class TinyVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape)
            )

        def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
            return x

    # Training step function
    def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        model.train()
        train_loss, train_acc = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y)
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    # Test step function
    def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                y_pred_class = torch.argmax(y_pred, dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    # Training loop
    def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int, device: torch.device):
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
            test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
            print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        return results

    #Setupinam random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    NUM_EPOCHS = 10

    # Paleidziam modeli
    model_0 = TinyVGG(input_shape=3, hidden_units=100, output_shape=10).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_0.parameters(), lr=0.0001)

    # Traininam
    results_0 = train(model=model_0, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS, device=device)
    print(results_0)

    #Pasikuriu naujo modelio instance (si karta naudosiu  transffer learning, imsiu Resnet50 pretrained modeli)
    model_1 = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    #Uzsaldom visus sluoksnius isksyrus paskutini
    #for param in model_1.parameters():
    #    param.requires_grad = False

    #Aprasom paskutini layeri
    model_1.fc = nn.Linear(2048,10)
    #Siunciam i devide
    model_1.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=0.0001)

    # Traininam resnet modeli
    results_1 = train(model=model_1, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS, device=device)
    print(results_1)

    #Apsirasom loss ir accuracy grafiku atvaizdavimo funkcija
    def plot_loss_curves(results: Dict[str,List[float]]):

        loss = results["train_loss"]
        test_loss = results["test_loss"]

        accuracy = results["train_acc"]
        test_accuracy = results["test_acc"]

        epochs = range(len(results["train_loss"]))

        plt.figure(figsize=(15,7))

        #Plot loss
        plt.subplot(1,2,1)
        plt.plot(epochs, loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        #Plot acc
        plt.subplot(1,2,2)
        plt.plot(epochs, accuracy, label="train_acc")
        plt.plot(epochs, test_accuracy, label="test_acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend();

    plot_loss_curves(results_0)
    plot_loss_curves(results_1)
if __name__ == "__main__":
    main()

