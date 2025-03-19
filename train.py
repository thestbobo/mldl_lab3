from models.customnet import CustomNet

import torch
import torch.nn as nn
from eval import validate, train
from data.dataloader import TinyImagenetDataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU NON disponibile, sto utilizzando la CPU!")

data_loader = TinyImagenetDataLoader(batch_size=64, num_workers=8, root_dir="tiny-imagenet-200")
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}:")
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')
