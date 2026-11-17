
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from data_loader import ECGDataset
from models import ECG_CNN

# Configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")

# Load datasets
train_dataset = ECGDataset("train")
val_dataset = ECGDataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Initialize model
model = ECG_CNN(num_classes=5).to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Save best model
Path("results").mkdir(exist_ok=True)
best_val_loss = float("inf")

print("Starting ECG CNN training...")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# Training loop
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for signals, labels in train_loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * signals.size(0)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * signals.size(0)

    val_loss /= len(val_dataset)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "results/best_cnn_model.pth"
        )

        print("Best model saved!")

print("\nTraining completed.")
print("Best validation loss:", round(best_val_loss, 4))
