import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1. Load Kaggle IMDB dataset CSV
# Replace 'IMDB Dataset.csv' with your actual CSV path
df = pd.read_csv('IMDB Dataset.csv.zip')  # Kaggle IMDB dataset

# -------------------------
# 2. Preprocess text and labels
texts = df['review'].values
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# -------------------------
# 3. Vectorize texts with TF-IDF (limit features for speed)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()
y = labels

# -------------------------
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -------------------------
# 5. Create DataLoader for batching
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -------------------------
# 6. Define the DEQ Module

class DEQModule(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(DEQModule, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Corrected input dimension
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, max_iter=30, tol=1e-4):
        z = torch.zeros(x.size(0), self.input_proj.out_features).to(x.device)
        x_proj = self.input_proj(x)

        for _ in range(max_iter):
            z_next = self.f(torch.cat([z, x_proj], dim=-1))
            if torch.norm(z_next - z) < tol:
                break
            z = z_next

        return z

# -------------------------
# 7. Define full DEQ sentiment classifier

class DEQClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DEQClassifier, self).__init__()
        self.deq = DEQModule(hidden_dim, input_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z_star = self.deq(x)
        logits = self.output_layer(z_star)
        return logits

# -------------------------
# 8. Training setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 5000
hidden_dim = 128
output_dim = 2  # positive/negative

model = DEQClassifier(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# 9. Training loop

def train_epoch():
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

    print(f"Train Loss: {total_loss/total_samples:.4f}, Accuracy: {total_correct/total_samples:.4f}")

# -------------------------
# 10. Evaluation function

def evaluate():
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
    print(f"Test Accuracy: {total_correct/total_samples:.4f}")

# -------------------------
# 11. Run training and evaluation

num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_epoch()
    evaluate()
