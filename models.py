import torch
import torch.nn as nn


class ECG_CNN(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":

    model = ECG_CNN()

    dummy_input = torch.randn(16, 12, 1000)

    output = model(dummy_input)

    print("Model:", model)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
