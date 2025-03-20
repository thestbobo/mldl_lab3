import torch


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
