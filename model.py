import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Here, the overall model of the dataset will be implemented - a neural network
# Smaller models will be used inside it to fill in missing values for different features
# The dataset that will be used is ml_dataset_tree.csv

#First convert the dataset into a dataframe
dataset = pd.read_csv("ml_dataset_tree.csv")

# Split the data from the targets
X = dataset.drop(columns=['target'])
y = dataset['target']

# Scaling numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Converting to tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test  = torch.tensor(y_test.values, dtype=torch.long)

# The definition of the actual Neural Network - is liable to change due to hyperparameter tuning
import torch.nn as nn

class StudentNN(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4)   # 4 output classes
        )

    def forward(self, x):
        return self.network(x)

# Creating the actual model
input_size = X_train.shape[1]
model = StudentNN(input_size)

# Loss and optimiser functions
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# The training of the network
epochs = 200

for epoch in range(epochs):

    # Forward pass
    outputs = model(X_train)

    loss = loss_function(outputs, y_train)

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluating metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

with torch.no_grad():

    outputs = model(X_test)

    # Predicted class index
    predictions = torch.argmax(outputs, dim=1)

# Convert tensors to numpy
y_true = y_test.numpy()
y_pred = predictions.numpy()

# Basic metrics

accuracy  = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted'
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion matrix

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Detailed class report

class_names = [
    'Pass',
    'Distinction',
    'Fail',
    'Withdrawn'
]

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)