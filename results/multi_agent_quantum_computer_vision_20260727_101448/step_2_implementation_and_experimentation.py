import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Set random seeds for deterministic execution and reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def generate_synthetic_image_dataset(num_samples=300):
    """
    Generates a synthetic binary classification dataset of 4x4 pixel images.
    Class 0: Horizontal pattern (rows) + noise
    Class 1: Vertical pattern (columns) + noise
    """
    images = []
    labels = []
    
    for _ in range(num_samples // 2):
        # Class 0: Horizontal feature
        img_h = np.zeros((4, 4), dtype=np.float32)
        row = np.random.randint(0, 4)
        img_h[row, :] = 1.0
        img_h += np.random.normal(0, 0.15, (4, 4)).astype(np.float32)
        images.append(img_h)
        labels.append(0)

        # Class 1: Vertical feature
        img_v = np.zeros((4, 4), dtype=np.float32)
        col = np.random.randint(0, 4)
        img_v[:, col] = 1.0
        img_v += np.random.normal(0, 0.15, (4, 4)).astype(np.float32)
        images.append(img_v)
        labels.append(1)

    X = np.array(images)[:, np.newaxis, :, :]  # Shape: (N, 1, 4, 4)
    y = np.array(labels, dtype=np.int64)
    return train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

# Circuit configuration
N_QUBITS = 4
N_LAYERS = 2
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    4-qubit Quantum Circuit:
    - AngleEncoding maps compressed spatial classical features into Hilbert Space.
    - StronglyEntanglingLayers applies trainable unitary transformations U(theta).
    - PauliZ expectation values measured on all qubits.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumClassicalHybridModel(nn.Module):
    """
    Hybrid Model Architecture:
    1. Classical Conv2D + Sigmoid downsamples 4x4 image (16 pixels) to 4 features.
    2. PennyLane Variational Quantum Circuit processes features in quantum state space.
    3. Classical Linear Layer maps quantum expectation values to class logits.
    """
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.classical_pre = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=2),  # 4x4 -> 2x2 = 4 elements
            nn.Flatten(),
            nn.Sigmoid()
        )
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.classical_post = nn.Sequential(
            nn.Linear(n_qubits, 2)
        )

    def forward(self, x):
        # Scale classical compressed features to [0, pi] for angle embedding
        compressed_features = self.classical_pre(x) * np.pi
        quantum_features = self.qlayer(compressed_features)
        logits = self.classical_post(quantum_features)
        return logits

def train_and_evaluate():
    print("==========================================================================")
    print(" QUANTUM-CLASSICAL HYBRID MODEL (QCNN / VQC) VERIFICATION & DIAGNOSTICS   ")
    print("==========================================================================")
    print(f"Backend Simulator Device: {dev.name} (Wires: {N_QUBITS})")
    
    # 1. Dataset Generation
    X_train, X_test, y_train, y_test = generate_synthetic_image_dataset(num_samples=300)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dataset summary: Train samples = {len(X_train)}, Test samples = {len(X_test)}")
    
    # 2. Model Initialization
    model = QuantumClassicalHybridModel(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    # Output Circuit Diagram
    dummy_features = torch.zeros(N_QUBITS)
    dummy_weights = torch.zeros(N_LAYERS, N_QUBITS, 3)
    print("\n--- Quantum Circuit Architecture ---")
    print(qml.draw(quantum_circuit)(dummy_features, dummy_weights))
    print("------------------------------------\n")

    # 3. Model Training
    print("Beginning Training Loop...")
    epochs = 20
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}%")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")

    # 4. Evaluation & Metrics Diagnostics
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='binary')
    rec = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    cm = confusion_matrix(all_targets, all_preds)

    print("\n==========================================================================")
    print(" PERFORMANCE EVALUATION METRICS                                          ")
    print("==========================================================================")
    print(f" Test Accuracy  : {acc * 100:.2f}%")
    print(f" Precision      : {prec:.4f}")
    print(f" Recall         : {rec:.4f}")
    print(f" F1-Score       : {f1:.4f}")
    print("\n Confusion Matrix:")
    print(cm)
    print("==========================================================================")
    
    # Validation assertion to ensure model learned meaningful patterns
    if acc > 0.70:
        print("\nSUCCESS: Convergence verified. Quantum-Classical model achieved >70% target accuracy.")
    else:
        print("\nWARNING: Model accuracy below target threshold. Convergence issue detected.")

if __name__ == "__main__":
    train_and_evaluate()