import os

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader


class TinyImagenetDataLoader:

    def __init__(self, root_dir, batch_size=64, num_workers=8):
        self.transform = T.Compose([T.Resize((64, 64)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                    ])
        self.train_transform = T.Compose([T.RandomResizedCrop(64),
                                          T.RandomHorizontalFlip(),
                                          T.RandomRotation(20),
                                          T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.root_dir = "dataset/tiny-imagenet-200"
        pass

    def get_val_loader(self):
        val_dataset = ImageFolder(root=os.path.join(self.root_dir, "val"),
                                  transform=self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        train_dataset = ImageFolder(root=os.path.join(self.root_dir, "train"),
                                    transform=self.train_transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

