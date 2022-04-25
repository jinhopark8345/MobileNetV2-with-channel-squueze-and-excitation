import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class CIFAR10(Dataset):
    def __init__(self, data_dir, transform=None, training=False):

        if training:
            self.data_dir = os.path.join(data_dir, "train")
        else:
            self.data_dir = os.path.join(data_dir, "test")

        # breakpoint()
        self.files = sorted(
            os.listdir(self.data_dir),
            key=lambda path: int(path.split("_", 1)[0]),
        )
        self.transform = transform
        classes = ["airplane" , "automobile" , "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.class2idx = {cls : idx for idx, cls in enumerate(classes)}


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cur_file = self.files[idx]
        img_path = os.path.join(self.data_dir, cur_file)
        img = Image.open(img_path)  # return PIL image
        label = cur_file.split("_", 1)[1].split(".")[0]
        target = self.class2idx[label]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def CIFAR10DataLoader(data_dir, batch_size, num_workers=0, shuffle=True, training=True):
    # transform for training
    if training:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # transform for test
    else:
        transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = CIFAR10(data_dir, transform=transform, training=training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def show_samples(dataloader):
    # get some random training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    print(f'{len(images), images[0], labels[0] = }')


def main():

    data_dir = "/home/jinho/ML-datasets/cifar/"
    sample_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    sample_loader = CIFAR10DataLoader(data_dir, batch_size=16)
    show_samples(sample_loader)


if __name__ == "__main__":
    main()
