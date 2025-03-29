import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tools import *  # Ensure this imports your EvaluatorDataset and any other required functions/classes

# =================== Model Creation ===================
class EvaluatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten the input (5, 8) into a vector of size 35
        self.flatten = nn.Flatten()
        # Define a simple fully connected network
        self.stack = nn.Sequential(
            nn.Linear(40, 32),
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
    # Set directory to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("============ Strand Evaluator ===========")

    # 1. Load the dataset
    dataset: EvaluatorDataset = torch.load("data/evaluator_dataset.pt", weights_only=False)
    dataset.summarise()
    print("1. Data loaded successfully")

    # Create a train/test split (e.g., 80% train, 20% test)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 2. Initialize Model
    model = EvaluatorModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("2. Model initialized successfully")

    # 3. Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("3. DataLoaders created successfully")

    # 4. Training Loop
    print("--- Training Model ---")
    num_epochs = 200
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, labels in train_loader:
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

        epoch_loss /= len(train_dataset)
        accuracy = correct / total

        # If loss is below threshold, show the metrics and break out of training
        if epoch_loss < 0.00001:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {accuracy:.5f}")
            print("--- Loss is below threshold of 0.00001, stopping training. ---")
            break

        if epoch % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {accuracy:.5f}")

    print("4. Model trained successfully")

    # Evaluate the model on the test set
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            test_loss += loss.item() * data.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_dataset)
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.5f}")

    # 5. Save the model
    torch.save(model.state_dict(), "models/evaluator_model.pth")
    print("5. Model saved successfully")

    # 6. Load the model (for demonstration)
    model.load_state_dict(torch.load("models/evaluator_model.pth"))
    print("6. Model loaded successfully")
