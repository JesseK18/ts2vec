import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =====================================================
# 1. Data Loading Function for UCR (with LabelEncoder)
# =====================================================
def load_ucr_dataset(dataset_name, root_dir="datasets/UCR"):
    """
    Loads a UCR dataset. Returns train_x, train_y, test_x, test_y
    with labels encoded to start from 0 and be consecutive.
    """
    train_file = os.path.join(root_dir, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(root_dir, dataset_name, f"{dataset_name}_TEST.tsv")

    train_data = np.loadtxt(train_file, delimiter="\t")
    test_data = np.loadtxt(test_file, delimiter="\t")

    # Separate labels and features
    train_y, train_x = train_data[:, 0], train_data[:, 1:]
    test_y, test_x = test_data[:, 0], test_data[:, 1:]

    # Use LabelEncoder to ensure labels are in [0 ... num_classes-1]
    all_labels = np.concatenate([train_y, test_y], axis=0)
    le = LabelEncoder()
    le.fit(all_labels)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    return train_x, train_y, test_x, test_y

# =====================================================
# 2. LSTM Classification Model
# =====================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        The final hidden state (from the last LSTM layer) is used for classification.
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1, :, :]  # shape (batch_size, hidden_dim)
        out = self.fc(last_hidden)   # shape (batch_size, num_classes)
        return out

# =====================================================
# 3. Training Function
# =====================================================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)                # shape: (batch_size, num_classes)
        loss = criterion(outputs, y_batch.long()) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# =====================================================
# 4. Evaluation Function
# =====================================================
def evaluate_model(model, test_loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(y_batch.numpy())

    accuracy = accuracy_score(trues, preds)
    return accuracy

# =====================================================
# 5. Main Script
# =====================================================
def main():
    # -----------------------------
    # 5.1 Load data
    # -----------------------------
    dataset_name = "OliveOil"  # Change to any other dataset in UCR
    train_x_np, train_y_np, test_x_np, test_y_np = load_ucr_dataset(dataset_name)

    # -----------------------------
    # 5.2 Convert to PyTorch Tensors
    #     Each input has shape (num_samples, seq_len, 1)
    # -----------------------------
    train_x_tensor = torch.tensor(train_x_np, dtype=torch.float32).unsqueeze(-1)
    train_y_tensor = torch.tensor(train_y_np, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x_np, dtype=torch.float32).unsqueeze(-1)
    test_y_tensor = torch.tensor(test_y_np, dtype=torch.long)

    # -----------------------------
    # 5.3 Create Dataloaders
    # -----------------------------
    batch_size = 32
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 5.4 Instantiate Model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Count the number of distinct classes in train_y_np
    num_classes = len(np.unique(train_y_np))
    print("Number of classes:", num_classes)

    model = LSTMClassifier(
        input_dim=1,         # We have 1 channel/feature for UCR data
        hidden_dim=64,       # LSTM hidden size
        num_layers=1,        # Try more layers if you'd like
        num_classes=num_classes
    ).to(device)

    # -----------------------------
    # 5.5 Loss & Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------------
    # 5.6 Training Loop
    # -----------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # -----------------------------
    # 5.7 Evaluate on Test Set
    # -----------------------------
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Classification Accuracy on {dataset_name}: {test_acc:.4f}")

if __name__ == "__main__":
    main()
