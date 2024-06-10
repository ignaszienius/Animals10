import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from torch import nn
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import os
from pathlib import Path
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm
from torchvision.transforms import ToPILImage

def main():

    #Pasidarau device GPU(cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Pasidarau dataseto direktorijos path
    data_path = Path("raw-img/")

    #Pasidarau funkcija pasizeti kas dataseto direktorijos viduje
    def dir_walk(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"Turime {len(dirnames)} direktorijas ir {len(filenames)} failus direktorijoje: {dirpath}")

    #Pasileidziam pasileidziam funkcija
    #dir_walk(data_path)

    #Pasidarau visu img path lista
    image_path_list = list(data_path.glob("*/*.*"))
    #print(image_path_list[:10])

    #Pasiziurim random image path
    random_image_path = random.choice(image_path_list)
    print(random_image_path)

    #pasidarom img class (kaip dir pavadinimas)
    image_class = random_image_path.parent.stem
    print(image_class)

    #Open img
    img = Image.open(random_image_path)
    img

    #Pasiprintinam visa metadata apie random img (path, class, dimensijas)
    print(f"Random image path: {random_image_path}")
    print(f"Random image class: {image_class}")
    print(f"Random image height: {img.height}")
    print(f"Random image width: {img.width}")

    #Paverciam img i tensorius, sutvarkom data

    data_transform = transforms.Compose([
        #Resize
        transforms.Resize(size=(64,64)),
        #Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        #Turn image to torch.Tensor
        transforms.ToTensor()
    ])
    data_transform(img)

    #Pasidarom atskira transform test datai
    test_transform = transforms.Compose([
        #Resize
        transforms.Resize(size=(64,64)),
        #Turn image to torch.Tensor
        transforms.ToTensor()
    ])


    #Pasiprintinam shape ir data type img
    #print(f"Img shape po transform: {data_transform(img).shape} ir data type {data_transform(img).dtype}")

    #Pasidarom funkcija perzeti dataseto img. Ju info, kaip atrodo pries ir po transform

    def plot_transformed_images(image_paths, transform, n=3):  # Corrected parameter name
        random_image_paths = random.sample(image_paths, k=n)
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots(nrows=1, ncols=2)  # Corrected 'ncol' to 'ncols'
                ax[0].imshow(f)
                ax[0].set_title(f"Original size:\n {f.size}")
                ax[0].axis(False)
                transformed_image = transform(f).permute(1, 2, 0)  # Corrected parameter name to 'transform'
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"Transformed Shape: \n {transformed_image.shape}")
                ax[1].axis(False)
                fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


    plot_transformed_images(image_paths=image_path_list,
                            transform=data_transform,
                            n=3
                            )

    #Pasidarom dataset 
    dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
    print(dataset)

    #Convertinam dataset i lista image (X) ir labels (y)
    X = []
    y = []

    for img, label in dataset:
        X.append(np.array(img))
        y.append(label)

    # paverciam listus i np arrays
    X = np.array(X)
    y = np.array(y)

    # Splitinam dataseta i train ir test setus
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Pasidarom CustomImageDataset klase
    class CustomImageDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
            self.to_pil = ToPILImage()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            image = self.to_pil(image)
            if self.transform:
                image = self.transform(image)
            return image, label
        
    #Pasidarom dataset instances
    train_dataset = CustomImageDataset(images=X_train, labels=y_train, transform=data_transform)
    test_dataset = CustomImageDataset(images=X_test, labels=y_test, transform=test_transform)

    # Pasiziurim shape kiekvieno seto
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    #Pasidarom class names list
    class_names = dataset.classes
    print(f"{class_names}")

    #Pasidarom dict is listo
    class_dict = dataset.class_to_idx
    print(f"{class_dict}")

    ### Dataseta train ir test loadinam i dataloaderi
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(train_dataloader)
    #print(test_dataloader)

    img_custom, label_custom = next(iter(train_dataloader))
    print(f"Vieno loaderio batcho shape: {img_custom.shape}")
    print(f"Label shape: {label_custom.shape}")


    #Pasidarom musu modelio class. Naudosiu TinyVGG modeli

    class TinyVGG(nn.Module):
        def __init__(self, input_shape:int,
                    hidden_units: int,
                    output_shape: int) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)   #default stride value same as kernel size
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)   #default stride value same as kernel size
            )
            self.classsifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*13*13,
                        out_features=output_shape)
            )
        def forward(self,x):
            x = self.conv_block_1(x)
            #print(x.shape)
            x = self.conv_block_2(x)
            #print(x.shape)
            x = self.classsifier(x)
            #print(x.shape)
            return x
    
    #Pasidarom train funkcija
    def train_step(model: torch.nn.Module,
                dataloader= torch.utils.data.DataLoader,
                loss_fn= torch.nn.Module,
                optimizer=torch.optim.Optimizer,
                device=device):
        model.train()
        train_loss, train_acc = 0,0
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class==y).sum().item()/len(y_pred)
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc  

    #Pasidarom test funkcija
    def test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device=device):
        model.eval()
        test_loss, test_acc = 0,  0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    #Pasaidarom funkcija kuri apjungs train step ir test step

    def train(model: torch.nn.Module,
            train_dataloader,
            test_dataloader,
            optimizer,
            loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
            epochs: int = 5,
            device=device):

    # 2. Create empty results dictionary
        results = {"train_loss": [],
                    "train_acc": [],
                    "test_loss": [],
                    "test_acc": []}

        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)

            # 4. Print out what's happening
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # 6. Return the filled results at the end of the epochs
        return results

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Epoch kiek
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_0 = TinyVGG(input_shape=3, # number of color channels of our target images
                    hidden_units=10,
                    output_shape=len(class_dict)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                lr=0.001)

    print(model_0)

if __name__ == "__main__":
    main()