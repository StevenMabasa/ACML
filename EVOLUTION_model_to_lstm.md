# Evolution from model.py to lstm_model.py

## Overview
This document explains the detailed transition from a simple feedforward neural network (`model.py`) to an advanced LSTM-based sequence model (`lstm_model.py`) for predicting student outcomes. The evolution reflects a shift from static feature-based classification to temporal sequence modeling that captures how student behavior evolves over time.

---

## Part 1: Problem Domain Shift

### model.py: Static Feature Classification
- **Data source**: `ml_dataset_tree.csv` - A flat dataset with static features
- **Approach**: Treat all features equally, no temporal information
- **Architecture**: Simple multi-layer perceptron (MLP)
  ```
  Input Features → Dense(64) → ReLU → Dense(32) → ReLU → Dense(4 classes)
  ```
- **Limitation**: Cannot capture how student engagement changes over time

### lstm_model.py: Temporal Sequence Modeling
- **Data source**: `sequence_dataset.npz` - A preprocessed dataset with:
  - **X_seq**: Daily sequences of student interactions (shape: 25562 students × 91 days × 3 features)
  - **X_static**: Static student attributes (shape: 25562 students × 3 features)
  - **y**: Target labels (Pass=0, Distinction=1, Fail=2, Withdrawn=3)
- **Approach**: Leverage temporal patterns in student behavior over the first 90 days
- **Architecture**: LSTM for sequences + static feature fusion
  ```
  X_seq (daily clicks, submissions, scores) → LSTM → Hidden state
  X_static (attempts, credits, reg date) → Dense(64) → ReLU → Dense(64)
  [Hidden state, Static features] → Concatenate → Dense(4 classes)
  ```
- **Advantage**: Captures behavioral trajectories, early warning signals

---

## Part 2: Data Preparation Pipeline

### model.py Data Flow
```
ml_dataset_tree.csv
    ↓
pd.read_csv()
    ↓
StandardScaler.fit_transform()
    ↓
train_test_split() [80/20, random_state=42]
    ↓
Convert to PyTorch tensors
    ↓
Model Training
```

### lstm_model.py Data Flow
```
Raw CSV files (studentInfo, studentVle, studentAssessment, etc.)
    ↓
sequence_preparation.py [Complex preprocessing]
    ├─ Filter to early prediction window (0-90 days)
    ├─ Build daily sequences for each student
    ├─ Extract 3 sequence features:
    │  ├─ VLE clicks per day
    │  ├─ Assessment submissions per day
    │  └─ Assessment scores per day
    ├─ Extract 3 static features:
    │  ├─ num_of_prev_attempts
    │  ├─ studied_credits
    │  └─ date_registration
    └─ Save as sequence_dataset.npz
    ↓
load_sequence_data() in lstm_model.py
    ├─ Handle NaN values (fill with column mean)
    ├─ Normalize X_seq: (X - mean) / std per channel
    ├─ Normalize X_static: (X - mean) / std per feature
    └─ Return normalized arrays
    ↓
build_dataloader()
    ├─ Stratified train_test_split() [80/20, stratify by y]
    ├─ Convert to PyTorch tensors
    ├─ Create TensorDataset instances
    ├─ Wrap in DataLoader (batch_size=128, shuffle=True for train)
    └─ Return (train_loader, test_loader)
    ↓
Model Training with batches
```

**Key differences**:
1. **Data source complexity**: model.py uses a single CSV; lstm_model.py merges 5 CSV files
2. **Temporal dimension**: lstm_model.py explicitly structures data with a time axis
3. **Feature engineering**: lstm_model.py creates time series from raw events; model.py assumes features are already engineered
4. **Data validation**: lstm_model.py handles NaN values explicitly; model.py assumes clean data

---

## Part 3: Architecture Evolution

### model.py: Simple MLP
```python
class StudentNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        return self.network(x)
```
- **Parameters**: ~Few thousand (depends on input_size)
- **Inductive bias**: None; treats each feature independently
- **Temporal capability**: None

### lstm_model.py: Sequence + Static Feature Fusion
```python
class LSTMClassifier(nn.Module):
    def __init__(self, seq_features, hidden_size=64, num_layers=1, 
                 static_features=0, num_classes=4, dropout=0.2):
        super().__init__()
        
        # Sequential component
        self.lstm = nn.LSTM(
            input_size=seq_features,           # 3 features per day
            hidden_size=hidden_size,            # 64
            num_layers=num_layers,              # 1
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Static component
        self.use_static = static_features > 0
        if self.use_static:
            self.static_fc = nn.Sequential(
                nn.Linear(static_features, hidden_size),    # 3 → 64
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.classifier = nn.Linear(hidden_size * 2, num_classes)  # 128 → 4
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)      # 64 → 4
    
    def forward(self, x_seq, x_static=None):
        # x_seq: (batch, 91, 3)
        _, (hn, _) = self.lstm(x_seq)
        hidden = hn[-1]  # (batch, 64)
        
        if self.use_static and x_static is not None:
            # x_static: (batch, 3)
            static = self.static_fc(x_static)  # (batch, 64)
            hidden = torch.cat([hidden, static], dim=1)  # (batch, 128)
        
        return self.classifier(hidden)  # (batch, 4)
```

**Key architectural differences**:
1. **LSTM layer**: Captures sequential dependencies across 91 days
2. **Dual pathways**:
   - Sequential: Daily engagement patterns
   - Static: Student background characteristics
3. **Fusion strategy**: Concatenation after separate processing
4. **Recurrent connections**: LSTM maintains hidden state across time steps (model.py has none)

---

## Part 4: Training Loop Evolution

### model.py Training
```python
for epoch in range(epochs):  # 200 epochs
    outputs = model(X_train)
    loss = loss_function(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

**Characteristics**:
- Single pass over entire training set per epoch (no mini-batches)
- No gradient clipping
- No weight initialization strategy
- No validation monitoring
- Logs every 10 epochs
- No device management (CPU assumed)

### lstm_model.py Training
```python
for epoch in range(1, EPOCHS + 1):  # 100 epochs
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')
```

**Within train_epoch()**:
```python
for x_seq, x_static, y in loader:  # Mini-batch iteration
    x_seq = x_seq.to(DEVICE)
    x_static = x_static.to(DEVICE)
    y = y.to(DEVICE)
    
    optimizer.zero_grad()
    logits = model(x_seq, x_static)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # ← NEW
    optimizer.step()
    
    total_loss += loss.item() * x_seq.size(0)
return total_loss / len(loader.dataset)
```

**Characteristics**:
- Mini-batch training (128 samples per batch)
- **Gradient clipping**: Prevents exploding gradients (critical for LSTMs)
- **Weight initialization**: 
  - Xavier uniform for input weights
  - Orthogonal for recurrent weights
  - Zero for biases
- **Validation monitoring**: Evaluates on test set every epoch
- **Device management**: GPU support
- **Per-epoch reporting**: Logs train and validation loss

---

## Part 5: Numerical Stability & Bug Fixes

### Issue Discovered: NaN Loss During Training

**Root Causes**:
1. **Missing values in data**: Column 2 of X_static (date_registration) contained NaN values
2. **Unnormalized data**: Raw features had vastly different scales (0-6988 for VLE clicks, 30-630 for credits)
3. **High learning rate**: 1e-3 was too aggressive for this architecture

### Solutions Implemented in lstm_model.py

#### 1. NaN Handling
```python
for col in range(X_static.shape[1]):
    col_data = X_static[:, col]
    nan_mask = np.isnan(col_data)
    if nan_mask.any():
        col_mean = np.nanmean(col_data)
        X_static[nan_mask, col] = col_mean
```
- Fills missing dates with column mean instead of propagating NaN through network

#### 2. Data Normalization
```python
# Normalize sequences
seq_mean = X_seq.mean(axis=(0, 1), keepdims=True)
seq_std = X_seq.std(axis=(0, 1), keepdims=True)
seq_std = np.maximum(seq_std, 1e-8)  # avoid division by zero
X_seq = (X_seq - seq_mean) / seq_std

# Normalize static features
static_mean = X_static.mean(axis=0, keepdims=True)
static_std = X_static.std(axis=0, keepdims=True)
static_std = np.maximum(static_std, 1e-8)
X_static = (X_static - static_mean) / static_std
```
- Brings all features to zero mean, unit variance
- Prevents some dimensions from dominating gradient computation

#### 3. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # GRAD_CLIP = 1.0
```
- Constrains gradient magnitude to prevent explosive updates (especially important for RNNs/LSTMs)

#### 4. Learning Rate Reduction
```
LR: 1e-3 → 1e-4  # 10x reduction
```
- Smaller steps allow for more stable convergence with normalized data

#### 5. Proper Weight Initialization
```python
for name, param in model.named_parameters():
    if 'weight_ih' in name:  # Input-to-hidden
        nn.init.xavier_uniform_(param)
    elif 'weight_hh' in name:  # Hidden-to-hidden (recurrent)
        nn.init.orthogonal_(param)
    elif 'bias' in name:
        nn.init.constant_(param, 0)
```
- Xavier/Glorot initialization: Keeps activations in reasonable range
- Orthogonal initialization for recurrent weights: Helps with gradient flow through time
- Zero bias: Standard practice

---

## Part 6: Evaluation Metrics

### model.py Metrics
```python
Accuracy, Precision, Recall, F1 Score (weighted)
Confusion Matrix
Classification Report
```

### lstm_model.py Metrics
```python
Same as model.py, plus:
- Per-epoch train and validation loss monitoring
- Early stopping potential (not implemented but infrastructure ready)
```

Both use the same sklearn metrics, but lstm_model.py provides more granular feedback during training.

---

## Part 7: Hyperparameter Comparison

| Parameter | model.py | lstm_model.py | Reasoning |
|-----------|----------|---------------|-----------|
| Epochs | 200 | 100 | 100 epochs sufficient with better data preprocessing |
| Learning Rate | 0.001 (1e-3) | 0.0001 (1e-4) | Lower for stability with normalized data |
| Batch Size | Full dataset | 128 | Mini-batches: better generalization, more stable gradients |
| Optimizer | Adam | Adam | Same |
| Gradient Clip | None | 1.0 | LSTM-specific: prevents exploding gradients |
| Dropout | None | 0.2 | LSTM-specific: regularization for small dataset |
| Hidden Size | 32 | 64 | Larger capacity needed for temporal modeling |

---

## Part 8: Conceptual Framework

### model.py: Assumption
**"Student outcomes depend only on static characteristics"**
- Previous attempts
- Credits studied
- Registration date

### lstm_model.py: Assumption
**"Student outcomes depend on both static characteristics AND how engagement evolves"**
- How clicking patterns change day-to-day
- When assessments are submitted
- Trajectory of assessment scores
- Interaction of behavioral patterns with student background

This is more realistic: two students with identical backgrounds can have different outcomes based on their engagement patterns.

---

## Part 9: Data Preprocessing Pipeline Details

### Before (model.py)
- Assumes features are pre-engineered in `ml_dataset_tree.csv`
- StandardScaler normalization
- No feature construction

### After (lstm_model.py)
Uses `sequence_preparation.py` which:
1. **Loads 5 raw CSV files** (event-level data)
2. **Filters to early prediction window** (0-90 days)
3. **Aggregates events into daily sequences**:
   - Sum of VLE clicks per day
   - Count of assessments per day
   - Mean assessment score per day
4. **Handles missing data** (fills with 0 for days with no events)
5. **Preserves student IDs** for traceability
6. **Saves compressed NPZ format** (space-efficient, fast loading)

This pipeline is domain-specific and captures the temporal nature of student learning data.

---

## Part 10: Why LSTM?

### Problem with MLP (model.py)
- Cannot distinguish between:
  - Student A: High engagement early, drops off
  - Student B: Low engagement early, increases
  
Both have same aggregate statistics, but different trajectories predict different outcomes.

### Solution with LSTM (lstm_model.py)
- **Hidden state**: Accumulates information across time steps
- **Forget gate**: Learns what old information to discard
- **Input gate**: Learns what new information to keep
- **Output gate**: Learns what hidden state to expose
- **Cell state**: Preserves long-term dependencies

LSTM can learn patterns like: "High initial engagement + subsequent withdrawal → likely Withdrawal outcome"

---

## Summary: The Transformation

| Aspect | model.py | lstm_model.py |
|--------|----------|---------------|
| **Data Model** | Flat features | Time series + static features |
| **Architecture** | MLP (3 layers) | LSTM + static fusion |
| **Training** | Full batch, 200 epochs | Mini-batch, 100 epochs, gradient clipping |
| **Preprocessing** | StandardScaler | Detailed pipeline with NaN handling |
| **Numerical Stability** | Basic | Robust (normalization, clipping, init) |
| **Inference Capability** | Single prediction | Sequence + context prediction |
| **Validation** | Final metrics only | Per-epoch monitoring |
| **Real-world applicability** | Early indicator | Early warning system (day 90 → predict final outcome) |

The evolution represents a fundamental shift from **static classification** to **dynamic sequence modeling**, enabling early detection of at-risk students based on behavioral trajectories rather than just background characteristics.
