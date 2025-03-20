import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
import os
from models.customnet import CustomNet  # Ensure this is your model file

# Initialize WandB for evaluation tracking
wandb.init(project="tiny-imagenet-evaluation", job_type="evaluation")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset path
dataset_path = os.path.abspath("dataset/tiny-imagenet-200/val")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure it matches training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = ImageFolder(root=dataset_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load trained model
model = CustomNet().to(device)
checkpoint_path = "checkpoints/best_model.pth"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded trained model for evaluation.")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

# Define loss function
criterion = nn.CrossEntropyLoss()

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

# Run evaluation
val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Log metrics to WandB
wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy})

# Finish WandB run
wandb.finish()


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # move data to gpu
            inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass: compute predictions
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to GPU if available
        inputs, targets = inputs.cuda(), targets.cuda()

        # Reset gradients to zero
        optimizer.zero_grad()

        # Forward pass, compute prediction
        outputs = model(inputs)

        # Compute the loss between predictions and true labels
        loss = criterion(outputs, targets)

        # Backpropagation: compute gradients
        loss.backward()

        # Update model parameters based on gradients
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_accuracy = 100. * correct / total
            print(f'Epoch [{epoch}] - Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {current_loss:.4f} | Acc: {current_accuracy:.2f}%')

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy
