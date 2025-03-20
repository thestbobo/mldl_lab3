from models.customnet import CustomNet

import torch
import wandb
import torch.nn as nn
from eval import validate, train
from data.dataloader import TinyImagenetDataLoader


wandb.init(project="tiny-imagenet-training", config={
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "optimizer": "SGD"
})

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU NON disponibile, sto utilizzando la CPU!")


data_loader = TinyImagenetDataLoader(batch_size=64, num_workers=8, root_dir="dataset/tiny-imagenet-200")
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(model.resnet.fc.parameters(), lr=0.001, weight_decay=1e-4)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}:")

    train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer)

    wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_acc})

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    wandb.log({"Epoch": epoch, "Validation Accuracy": val_accuracy})

    # Best validation accuracy
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        best_model_path = 'checkpoints/best_model.pth'
        torch.save(model.state_dict(), best_model_path)

        # Upload model to WandB
        wandb.save(best_model_path)


print(f'Best validation accuracy: {best_acc:.2f}%')
