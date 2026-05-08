import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = 'sequence_dataset.npz'
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
HIDDEN_SIZE = 64
GRAD_CLIP = 1.0
NUM_LAYERS = 1
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sequence_data(path):
    archive = np.load(path, allow_pickle=True)
    X_seq = archive['X_seq'].astype(np.float32)
    X_static = archive['X_static'].astype(np.float32)
    y = archive['y'].astype(np.int64)
    
    # Handle NaN values in X_static
    for col in range(X_static.shape[1]):
        col_data = X_static[:, col]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            col_mean = np.nanmean(col_data)
            X_static[nan_mask, col] = col_mean
    
    # Normalize X_seq (sequences of interactions)
    seq_mean = X_seq.mean(axis=(0, 1), keepdims=True)
    seq_std = X_seq.std(axis=(0, 1), keepdims=True)
    seq_std = np.maximum(seq_std, 1e-8)  # avoid division by zero
    X_seq = (X_seq - seq_mean) / seq_std
    
    # Normalize X_static (static features)
    static_mean = X_static.mean(axis=0, keepdims=True)
    static_std = X_static.std(axis=0, keepdims=True)
    static_std = np.maximum(static_std, 1e-8)
    X_static = (X_static - static_mean) / static_std
    
    return X_seq, X_static, y


class LSTMClassifier(nn.Module):
    def __init__(self, seq_features, hidden_size=64, num_layers=1, static_features=0, num_classes=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.use_static = static_features > 0
        if self.use_static:
            self.static_fc = nn.Sequential(
                nn.Linear(static_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x_seq, x_static=None):
        _, (hn, _) = self.lstm(x_seq)
        hidden = hn[-1]
        if self.use_static and x_static is not None:
            static = self.static_fc(x_static)
            hidden = torch.cat([hidden, static], dim=1)
        return self.classifier(hidden)


def build_dataloader(X_seq, X_static, y, batch_size=128, test_size=0.2, random_state=42):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_seq_train = torch.from_numpy(X_seq[train_idx])
    X_static_train = torch.from_numpy(X_static[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_seq_test = torch.from_numpy(X_seq[test_idx])
    X_static_test = torch.from_numpy(X_static[test_idx])
    y_test = torch.from_numpy(y[test_idx])

    train_dataset = TensorDataset(X_seq_train, X_static_train, y_train)
    test_dataset = TensorDataset(X_seq_test, X_static_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x_seq, x_static, y in loader:
        x_seq = x_seq.to(DEVICE)
        x_static = x_static.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_seq, x_static)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for x_seq, x_static, y in loader:
            x_seq = x_seq.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x_seq, x_static)
            loss = criterion(logits, y)
            total_loss += loss.item() * x_seq.size(0)
            preds.append(torch.argmax(logits, dim=1).cpu())
            targets.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return total_loss / len(loader.dataset), y_true, y_pred


def main():
    print(f'Loading sequence data from {DATA_PATH}')
    X_seq, X_static, y = load_sequence_data(DATA_PATH)
    print('X_seq shape:', X_seq.shape)
    print('X_static shape:', X_static.shape)
    print('y shape:', y.shape)

    train_loader, test_loader = build_dataloader(X_seq, X_static, y, batch_size=BATCH_SIZE)
    seq_features = X_seq.shape[2]
    static_features = X_static.shape[1]

    model = LSTMClassifier(
        seq_features=seq_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        static_features=static_features,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'Using device: {DEVICE}')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, y_true, y_pred = evaluate(model, test_loader, criterion)
        print(f'Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    print('\nFinal evaluation:')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['Pass', 'Distinction', 'Fail', 'Withdrawn'], zero_division=0))


if __name__ == '__main__':
    main()
