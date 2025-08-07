import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

# PyTorch-based autoencoder for anomaly detection.
# Train on normal/benign only; score = reconstruction error; classify by threshold.

class AutoencoderNet(nn.Module):
    """PyTorch Autoencoder Network"""
    
    def __init__(self, input_dim, hidden_dims=(128, 64), latent_dim=32):
        super(AutoencoderNet, self).__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for reconstruction)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SklearnAutoencoder(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=(128, 64), latent_dim=32, epochs=20, batch_size=256, lr=1e-3, threshold=None):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y=None):
        # Train only on "normal" / benign
        if y is not None:
            # mask = (y == 0) | (y == 'normal') | (y == 'Normal') | (y == 'BENIGN') | (y == 'benign')
            # X_train = X[mask] if hasattr(X, '__getitem__') else X.loc[mask]
            if "Worms" in y.unique():
                cats = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
                y = y.map(lambda x:x if isinstance (x, int) else cats.index(x))
        X_train = X
        
        n_cls = len(np.unique(y)) if y is not None else 1
        # Convert to numpy array if pandas DataFrame
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        
        # Ensure we have data
        if len(X_train) == 0:
            raise ValueError("No training data available after filtering for normal/benign samples")
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        input_dim = X_train_scaled.shape[1]
        self.input_dim = input_dim
        self.model = AutoencoderNet(input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Input and target are the same for autoencoder
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_train)), shuffle=True)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}')
        
        classification_model = nn.Sequential(
            self.model.encoder,
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_cls)
        ).to(self.device)
        self.n_cls = n_cls
        self.model = classification_model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classification_model.parameters(), lr=self.lr, weight_decay=1e-5)
        print("Autoencoder model trained successfully")
        print("Training classification model with encoder backbone...")
        dataset = TensorDataset(X_tensor, torch.as_tensor(y, device=self.device))  
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_train)), shuffle=True)
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits,batch_target.long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}')
        
        # Determine threshold on training set (95th percentile)
        self.model.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            errors = []
            accuracy = []
            batch_size = min(1000, len(X_train_scaled))
            for i in range(0, len(X_train_scaled), batch_size):
                batch_data = X_train_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                y_batch = y[i:i+batch_size]
                y_tensor = torch.FloatTensor(np.array(y_batch)).to(self.device)
                logits = self.model(batch_tensor)
                loss = criterion(logits, y_tensor.reshape(-1).long())
                errors.append(loss.cpu().numpy())
                acc = torch.mean((logits.argmax(dim=1) == y_tensor).float()).item()
                accuracy.append(acc)
            errors = np.array(errors).flatten()
            accuracy = np.mean(accuracy)
        
        print(f"Training accuracy: {accuracy:.4f}")
        print(f"Average training loss: {np.mean(errors):.4f}, std: {np.std(errors):.4f}")
        return self

    def predict(self, X):
        logits = self.model(torch.FloatTensor(self.scaler.transform(X)).to(self.device))
        predictions = logits.argmax(dim=1).cpu().numpy()
        return predictions

    def predict_proba(self, X):
        logits = self.model(torch.FloatTensor(self.scaler.transform(X)).to(self.device))
        probabilities = torch.softmax(logits, dim=1).cpu().detach().numpy()
        return probabilities
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'num_classes': self.n_cls if hasattr(self, 'n_cls') else None
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore parameters
        self.hidden_dims = checkpoint['hidden_dims']
        self.latent_dim = checkpoint['latent_dim']
        self.threshold = checkpoint['threshold']
        self.scaler = checkpoint['scaler']
        
        # Recreate and load model
        input_dim = checkpoint['input_dim']
        self.model = AutoencoderNet(input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        self.model = nn.Sequential(
            self.model.encoder,
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64,checkpoint['num_classes'])
            ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        return self
