import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

from dataset import GuitarDataSet


class GuitarNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(188, 100),
            nn.Tanh(),
            nn.Linear(100, 94),
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

wav_paths = []
label_paths = []

train_wavs, val_wavs, train_labels, val_labels = train_test_split(
    wav_paths,
    label_paths,
    test_size=0.2,
    random_state=42
) 

train_dataset = GuitarDataSet(train_wavs, train_labels)
val_dataset = GuitarDataSet(val_wavs, val_labels)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


model = GuitarNN()

criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def run_training():
    epochs = 50

    for epoch in range (epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.squeeze(0)
            targets = targets.squeeze(0)

            logits = model(input)
            loss = criteria(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.squeeze(0)
                    val_targets = val_targets.squeeze(0)

                    val_logits = model(val_inputs)
                    val_loss = criteria(val_logits, val_targets)

                    running_val_loss += val_loss.item()

            avg_train = running_loss / len(train_loader)
            avg_val = running_val_loss / len(val_loader)

            print(f"Epoch {epoch:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")