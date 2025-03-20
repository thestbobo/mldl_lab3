import torch
from torch import nn
import torchvision.models as models

# Define the custom neural network


class CustomNet(nn.Module):

    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.resnet.parameters():
            param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    # def __init__(self, num_classes=200):
    #     super(CustomNet, self).__init__()
    #     # Define layers of the neural network
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    #     self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    #     self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    #     self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    #
    #     # ACTIVATION layer
    #     self.relu = nn.ReLU()
    #
    #     # POOLING layer
    #
    #     self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #     # Add more layers...
    #     self.fc1 = nn.Linear(256 * 8 * 8, 512)
    #     self.fc2 = nn.Linear(512, num_classes)
    #
    # def forward(self, x):
    #     # Define forward pass
    #
    #     # INPUT: B x 3 x 224 x 224
    #
    #     x = self.relu(self.conv1(x))  # B x 64 x 224 x 224
    #     x = self.relu(self.conv2(x))  # B x 128 x 224 x 224
    #
    #     x = self.max_pool(x)  # B x 128 x 112 x 112
    #
    #     x = self.relu(self.conv3(x))  # B x 256 x 112 x 112
    #     x = self.relu(self.conv4(x))  # B x 256 x 112 x 112
    #
    #     x = self.max_pool(x)  # B x 256 x 56 x 56
    #
    #     x = self.relu(self.conv5(x))  # B x 512 x 56 x 56
    #     x = self.relu(self.conv6(x))  # B x 512 x 56 x 56
    #     x = self.relu(self.conv7(x))  # B x 512 x 56 x 56
    #
    #     x = self.max_pool(x) # B x 512 x 28 x 28
    #
    #     x = torch.flatten(x, start_dim=1)  # Flatten to (B x 512*28*28
    #     x = self.relu(self.fc1(x))  # B x 200
    #
    #     x = self.fc2(x)    # B x num_classes
    #
    #
    #     return x
