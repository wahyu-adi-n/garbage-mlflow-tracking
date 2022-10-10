import os
import pathlib
import zipfile
import splitfolders as spf
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path


class GarbageCustomDataset(Dataset):
    def __init__(self, target_dir,  transform=None):
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

    def __len__(self):
        return len(self.paths)

    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)


def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def create_dir_extract():
    data_path = Path('data/')
    image_path = data_path / "dataset"
    if image_path.is_dir():
        print(f"{image_path} directory exists!")
    else:
        print(f"{image_path} did not exists, creating one!")
        image_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_path / "garbage.zip", "r") as zip_reff:
            print('Unzipping file data!')
            zip_reff.extractall(image_path)


def split_data(input_dir, out_dir, seed=42):
    spf.ratio(input=input_dir,
              output=out_dir,
              seed=seed,
              ratio=(0.8, 0.1, 0.1),
              group_prefix=None)


def augment_data(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
    train_transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.CenterCrop(size=(224, 224)),
        T.RandomHorizontalFlip(0.25),
        T.RandomVerticalFlip(0.25),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    val_transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.CenterCrop(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    test_transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.CenterCrop(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform, test_transform


def dataloaders(train_dir: str,
                val_dir: str,
                test_dir: str,
                batch_size: int,
                test_batch_size: int):
    train_transform, val_transform, test_transform = augment_data()
    train_data = GarbageCustomDataset(target_dir=train_dir,
                                      transform=train_transform)

    val_data = GarbageCustomDataset(target_dir=val_dir,
                                    transform=test_transform)

    test_data = GarbageCustomDataset(target_dir=test_dir,
                                     transform=test_transform)

    train_dl = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          num_workers=os.cpu_count(),
                          shuffle=True)

    val_dl = DataLoader(dataset=val_data,
                        batch_size=test_batch_size,
                        num_workers=os.cpu_count(),
                        shuffle=False)

    test_dl = DataLoader(dataset=test_data,
                         batch_size=test_batch_size,
                         num_workers=os.cpu_count(),
                         shuffle=False)

    return train_dl, val_dl, test_dl
