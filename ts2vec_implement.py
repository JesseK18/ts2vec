import os
import numpy as np
import torch
from ts2vec import TS2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load a UCR dataset from .tsv files
def load_ucr_dataset(dataset_name, root_dir="datasets/UCR"):
    train_file = os.path.join(root_dir, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(root_dir, dataset_name, f"{dataset_name}_TEST.tsv")

    # Load data using numpy (tab-separated values)
    train_data = np.loadtxt(train_file, delimiter="\t")
    test_data = np.loadtxt(test_file, delimiter="\t")

    # Separate labels and features
    train_y, train_x = train_data[:, 0], train_data[:, 1:]
    test_y, test_x = test_data[:, 0], test_data[:, 1:]

    # Normalize labels to start from 0 (important for classification)
    train_y -= train_y.min()
    test_y -= test_y.min()

    return train_x, train_y, test_x, test_y

# Load the UCR dataset
dataset_name = "EthanolLevel"  # Change to any dataset in UCR
train_x_np, train_y, test_x_np, test_y = load_ucr_dataset(dataset_name)

# Convert data to PyTorch tensors and add an extra dimension for the channel/feature.
# This converts the shape from (num_samples, sequence_length) to (num_samples, sequence_length, 1)
train_x_tensor = torch.tensor(train_x_np, dtype=torch.float32).unsqueeze(-1)
test_x_tensor = torch.tensor(test_x_np, dtype=torch.float32).unsqueeze(-1)

# Setup device and initialize TS2Vec.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TS2Vec(input_dims=train_x_tensor.shape[-1], device=device)

# For training, convert the training tensor back to a NumPy array.
train_x_for_fit = train_x_tensor.cpu().numpy()
model.fit(train_x_for_fit, n_epochs=10, verbose=True)

# For encoding, also convert the tensors to NumPy arrays.
# This will produce a representation of shape (n_samples, time_steps, feature_dim)
train_repr = model.encode(train_x_tensor.cpu().numpy())
test_repr = model.encode(test_x_tensor.cpu().numpy())

# Aggregate the time dimension (axis=1) by taking the mean, so that the representations become 2D.
train_repr = train_repr.mean(axis=1)
test_repr = test_repr.mean(axis=1)

# Train a classifier (Logistic Regression)
clf = RandomForestClassifier(n_estimators=100, random_state=42) # LogisticRegression()
clf.fit(train_repr, train_y)

# Evaluate on the test set
preds = clf.predict(test_repr)
accuracy = accuracy_score(test_y, preds)
print(f"Classification Accuracy on {dataset_name}: {accuracy:.4f}")
