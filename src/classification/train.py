import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Ejecuta una época de entrenamiento.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item() * images.size(0)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()



    epoch_loss = running_loss / total

    epoch_acc = correct / total


    return epoch_loss, epoch_acc

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item() *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()



    epoch_loss = running_loss / total

    epoch_accuracy = correct / total


    return epoch_loss, epoch_accuracy

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path
):

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }


    for epoch in range(epochs):

        print(
            f"Epoch {epoch+1}/{epochs}"
        )


        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )


        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )


        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )


        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


        if val_loss < best_val_loss:

            best_val_loss = val_loss


            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                "Modelo guardado"
            )


    return history