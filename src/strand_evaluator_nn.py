import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tools import *

# import dataset from data/evaluator_dataset.pt


# =================== Model Creation ===================
class EvaluatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten the input (5, 7) into a vector of size 35
        self.flatten = nn.Flatten()
        # Define a simple fully connected network
        self.stack = nn.Sequential(
            nn.Linear(35, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output a single value for binary classification
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.stack(x)
        return x


if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("============ Strand Evaluator ===========")

    # 1. Load the dataset
    dataset: EvaluatorDataset = torch.load(
        "data/evaluator_dataset.pt", weights_only=False
    )
    dataset.summarise()
    print("1. Data loaded successfully")

    # 2. Initialize Model
    model = EvaluatorModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("2. Model initialized successfully")

    # 3. Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("3. DataLoader created successfully")

    # 4. Training Loop

    print("--- Training Model ---")
    num_epochs = 200

    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Reshape labels to match output shape and convert to float
            labels = labels.unsqueeze(1).float()

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)

            # Compute predictions: apply sigmoid to convert logits to probabilities,
            # then threshold at 0.5 to get binary predictions.
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss /= len(dataset)
        accuracy = correct / total

        # If loss is below threshold, show the metrics and break out of training
        if epoch_loss < 0.0001:
            print(
                f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {accuracy:.5f}"
            )
            print("--- Loss is below threshold of 0.0001, stopping training. ---")
            break

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {accuracy:.5f}"
            )

    print("4. Model trained successfully")

    # 5. Save the model
    torch.save(model.state_dict(), "models/evaluator_model.pth")
    print("5. Model saved successfully")
    # 6. Load the model
    model.load_state_dict(torch.load("models/evaluator_model.pth"))
    print("6. Model loaded successfully")
