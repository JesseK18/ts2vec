import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
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
        last_hidden = h_n[-1, :, :]  # shape: (batch_size, hidden_dim)
        out = self.fc(last_hidden)   # shape: (batch_size, num_classes)
        return out

# =====================================================
# 3. Adversarial Attack Functions (for test-time noise)
# =====================================================
def fgsm_attack(model, loss_fn, x, y, epsilon):
    """Generates adversarial examples using the Fast Gradient Sign Method (FGSM)."""
    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = loss_fn(outputs, y)
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + epsilon * grad_sign
    return x_adv.detach()

def bim_attack(model, loss_fn, x, y, epsilon, alpha, iters):
    """
    Generates adversarial examples using the Basic Iterative Method (BIM),
    an iterative version of FGSM.
    """
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + alpha * grad_sign
        # Project perturbations so that we remain within the epsilon ball of the original x
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.detach()
    return x_adv

def pgd_attack(model, loss_fn, x, y, epsilon, alpha, iters):
    """
    Generates adversarial examples using Projected Gradient Descent (PGD),
    sometimes also called PGM. A random start within the epsilon ball is used.
    """
    # Random initialization within the epsilon ball
    x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = x_adv.detach()
    for _ in range(iters):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + alpha * grad_sign
        # Project back into the epsilon-ball around the original x
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.detach()
    return x_adv

# =====================================================
# 4. Training Function (using clean data)
# =====================================================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
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
# 5. Evaluation Function (injecting adversarial noise on test set)
# =====================================================
def evaluate_model(model, test_loader, device, attack_type=None, epsilon=0.1, alpha=0.01, iters=10):
    """
    Evaluates the model on the test set. If attack_type is specified ('fgsm', 'bim', or 'pgd'/'pgm'),
    adversarial noise is injected into the test examples before classification.
    """
    model.eval()
    preds = []
    trues = []
    loss_fn = nn.CrossEntropyLoss()

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Only add adversarial noise on test examples if requested
        if attack_type == 'fgsm':
            x_batch_adv = fgsm_attack(model, loss_fn, x_batch, y_batch, epsilon)
        elif attack_type == 'bim':
            x_batch_adv = bim_attack(model, loss_fn, x_batch, y_batch, epsilon, alpha, iters)
        elif attack_type in ['pgd', 'pgm']:
            x_batch_adv = pgd_attack(model, loss_fn, x_batch, y_batch, epsilon, alpha, iters)
        else:
            x_batch_adv = x_batch

        # Run the model on (potentially perturbed) test data
        with torch.no_grad():
            outputs = model(x_batch_adv)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(trues, preds)
    return accuracy

# =====================================================
# 6. Main Script
# =====================================================
def main():
    # 6.1 Load data
    dataset_name = ""  # Change this to any available UCR dataset
    train_x_np, train_y_np, test_x_np, test_y_np = load_ucr_dataset(dataset_name)

    # 6.2 Convert to PyTorch Tensors
    # Each input is reshaped to (num_samples, seq_len, 1)
    train_x_tensor = torch.tensor(train_x_np, dtype=torch.float32).unsqueeze(-1)
    train_y_tensor = torch.tensor(train_y_np, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x_np, dtype=torch.float32).unsqueeze(-1)
    test_y_tensor = torch.tensor(test_y_np, dtype=torch.long)

    # 6.3 Create Dataloaders
    batch_size = 32
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6.4 Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(train_y_np))
    print("Number of classes:", num_classes)

    model = LSTMClassifier(
        input_dim=1,         # UCR datasets have 1 channel/feature
        hidden_dim=64,       # LSTM hidden size (can be adjusted)
        num_layers=1,        # Number of LSTM layers (can be increased)
        num_classes=num_classes
    ).to(device)

    # 6.5 Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6.6 Training Loop (training on clean data)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # 6.7 Evaluate on Test Set with Adversarial Noise
    # Set attack_type to 'fgsm', 'bim', or 'pgd' (or 'pgm') to inject noise into test samples.
    # Use attack_type = None for clean test evaluation.
    attack_type = 'fgsm'  # For example, applying the BIM attack on test data
    epsilon = 0.1        # Maximum perturbation
    alpha = 0.01         # Step size for iterative attacks
    iters = 10           # Number of iterations for iterative attacks

    test_acc_adv = evaluate_model(model, test_loader, device,
                                  attack_type=attack_type,
                                  epsilon=epsilon,
                                  alpha=alpha,
                                  iters=iters)
    print(f"Adversarial Classification Accuracy on {dataset_name} with {attack_type} attack: {test_acc_adv:.4f}")

if __name__ == "__main__":
    main()
