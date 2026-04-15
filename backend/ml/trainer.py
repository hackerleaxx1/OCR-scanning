"""
Model Trainer for Vendor Invoice Neural Networks.

Handles training and weight persistence for PyTorch models specific to each vendor.
Implements the full training pipeline: data loading, vectorization, training loop,
weight saving/loading to CSV format.

Key features:
- Per-vendor training (each vendor has independent model)
- Vocabulary building from training texts
- CSV-based weight persistence (vendor-friendly, human-readable)
- Fine-tuning support with lower learning rate
- Training metadata tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json

from ml.model import VendorInvoiceModel, get_model_config
from ocr.text_processor import normalize_text, load_vocabulary, build_vocabulary


class ModelTrainer:
    """
    Handles training and weight persistence for vendor models.

    Manages the full training lifecycle:
    1. Load or prepare training data
    2. Build/update vocabulary from training texts
    3. Convert data to PyTorch tensors
    4. Train model with specified hyperparameters
    5. Save weights and metadata

    Weights are stored in CSV format for easy inspection and portability.
    """

    def __init__(self, vendor_id: str, vendor_dir: Path, config: dict = None):
        """
        Initialize trainer for a specific vendor.

        Args:
            vendor_id: Unique vendor identifier
            vendor_dir: Path to vendor's storage directory
            config: Optional override for default model config
        """
        self.vendor_id = vendor_id
        self.vendor_dir = Path(vendor_dir)
        self.vendor_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or get_model_config()
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Metadata for tracking training state
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """
        Load training metadata or create default.

        Metadata tracks:
        - Training status (untrained, trained)
        - Number of training samples
        - Last trained timestamp
        - Model accuracy on training set
        """
        meta_path = self.vendor_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {
            'vendor_id': self.vendor_id,
            'status': 'untrained',
            'training_samples': 0,
            'last_trained': None,
            'accuracy': None
        }

    def _save_metadata(self):
        """Save training metadata to disk."""
        meta_path = self.vendor_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f)

    def _load_training_data(self) -> pd.DataFrame or None:
        """
        Load training data from vendor's CSV file.

        Returns:
            DataFrame with training samples, or None if file doesn't exist
        """
        csv_path = self.vendor_dir / "training_data.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)

    def _prepare_training_data(self, df: pd.DataFrame):
        """
        Convert DataFrame to PyTorch tensors for training.

        1. Vectorize OCR texts using TF-IDF
        2. Encode categorical labels (invoice_number, date, item_name)
        3. Prepare regression targets (amounts, quantities, prices)

        Args:
            df: DataFrame with training samples

        Returns:
            Tuple of (X_tensor, y_tensor dict)
        """
        vectorizer = load_vocabulary(self.vendor_id, self.vendor_dir)
        if vectorizer is None:
            raise ValueError(f"No vocabulary found for vendor {self.vendor_id}")

        # Convert OCR texts to TF-IDF feature vectors
        texts = df['ocr_text'].tolist()
        X = np.array([vectorizer.transform([normalize_text(t)]).toarray()[0] for t in texts])

        # Encode categorical columns to integer indices
        for col, cat_col in [('invoice_number', 'invoice_number_enc'),
                              ('invoice_date', 'invoice_date_enc'),
                              ('item_name', 'item_name_enc')]:
            df[col] = df[col].astype('category')
            df[f'{col}_enc'] = df[col].cat.codes

        invoice_numbers = df['invoice_number_enc'].values
        invoice_dates = df['invoice_date_enc'].values
        item_names = df['item_name_enc'].values

        # Regression targets
        invoice_amounts = df['invoice_amount'].astype(float).values
        item_quantities = df['item_quantity'].astype(float).values
        per_item_prices = df['per_item_price'].astype(float).values
        total_prices = df['total_item_price'].astype(float).values

        # Build label encoders for metadata (maps index back to original value)
        self.metadata['label_encoders'] = {
            'invoice_numbers': {i: c for i, c in enumerate(df['invoice_number'].astype('category').cat.categories)},
            'invoice_dates': {i: c for i, c in enumerate(df['invoice_date'].astype('category').cat.categories)},
            'item_names': {i: c for i, c in enumerate(df['item_name'].astype('category').cat.categories)},
        }

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = {
            'invoice_number': torch.LongTensor(invoice_numbers),
            'invoice_date': torch.LongTensor(invoice_dates),
            'item_name': torch.LongTensor(item_names),
            'invoice_amount': torch.FloatTensor(invoice_amounts),
            'item_quantity': torch.FloatTensor(item_quantities),
            'per_item_price': torch.FloatTensor(per_item_prices),
            'total_item_price': torch.FloatTensor(total_prices)
        }

        return X_tensor, y_tensor

    def train(self, epochs: int = 50, learning_rate: float = 0.01,
              fine_tune: bool = False, batch_size: int = 16) -> dict:
        """
        Train the vendor model on accumulated data.

        Args:
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            fine_tune: If True, load existing weights and use lower learning rate
            batch_size: Training batch size

        Returns:
            Dict with training results (status, epochs, accuracy, samples)
        """
        df = self._load_training_data()
        if df is None or len(df) < 2:
            return {'status': 'error', 'message': 'Not enough training data'}

        # Rebuild vocabulary with all training texts
        texts = df['ocr_text'].tolist()
        vectorizer = build_vocabulary(texts, self.vendor_id, self.vendor_dir,
                                        max_features=self.config['input_dim'])

        # Update config with actual label counts
        actual_input_dim = len(vectorizer.vocabulary_)
        self.config['num_invoice_numbers'] = len(df['invoice_number'].unique())
        self.config['num_dates'] = len(df['invoice_date'].unique())
        self.config['num_items'] = len(df['item_name'].unique())

        X, y = self._prepare_training_data(df)

        # Create model with actual dimensions
        model = VendorInvoiceModel(
            input_dim=actual_input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout=self.config['dropout'],
            num_invoice_numbers=self.config['num_invoice_numbers'],
            num_dates=self.config['num_dates'],
            num_items=self.config['num_items']
        )
        model.to(self.device)

        # Load existing weights if fine-tuning
        if fine_tune:
            self._load_weights(model)

        # Setup optimizer with optional weight decay for regularization
        lr = self.config.get('fine_tune_learning_rate', 0.001) if fine_tune else learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # Reduce learning rate when training plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Create data loader
        dataset = TensorDataset(X, y['invoice_number'], y['invoice_date'],
                                 y['item_name'], y['invoice_amount'],
                                 y['item_quantity'], y['per_item_price'],
                                 y['total_item_price'])

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss functions - CrossEntropy for classification, MSE for regression
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                y_inv_num = batch[1].to(self.device)
                y_inv_date = batch[2].to(self.device)
                y_item = batch[3].to(self.device)
                y_amount = batch[4].to(self.device)
                y_qty = batch[5].to(self.device)
                y_price = batch[6].to(self.device)
                y_total = batch[7].to(self.device)

                optimizer.zero_grad()

                outputs = model(X_batch)

                # Classification loss (higher weight for classification tasks)
                cls_loss = (criterion_cls(outputs['invoice_number'], y_inv_num) +
                           criterion_cls(outputs['invoice_date'], y_inv_date) +
                           criterion_cls(outputs['item_name'], y_item))
                # Regression loss
                reg_loss = (criterion_reg(outputs['invoice_amount'], y_amount) +
                           criterion_reg(outputs['item_quantity'], y_qty) +
                           criterion_reg(outputs['per_item_price'], y_price) +
                           criterion_reg(outputs['total_item_price'], y_total))

                # Combined loss with configurable weights
                loss = self.config.get('classification_weight', 2.0) * cls_loss + \
                       self.config.get('regression_weight', 1.0) * reg_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step(total_loss)

        # Save weights and metadata
        self._save_weights(model)
        self._save_metadata()

        # Calculate accuracy on training set (for information only)
        model.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            outputs = model(X_device)

            inv_num_acc = (torch.argmax(outputs['invoice_number'], dim=1) ==
                          y['invoice_number'].to(self.device)).float().mean().item()
            inv_date_acc = (torch.argmax(outputs['invoice_date'], dim=1) ==
                           y['invoice_date'].to(self.device)).float().mean().item()
            item_acc = (torch.argmax(outputs['item_name'], dim=1) ==
                       y['item_name'].to(self.device)).float().mean().item()

        accuracy = (inv_num_acc + inv_date_acc + item_acc) / 3

        self.metadata['status'] = 'trained'
        self.metadata['last_trained'] = pd.Timestamp.now().isoformat()
        self.metadata['accuracy'] = accuracy
        self._save_metadata()

        return {
            'status': 'trained',
            'epochs': epochs,
            'accuracy': accuracy,
            'samples': len(df)
        }

    def _save_weights(self, model: torch.nn.Module):
        """
        Save model weights and biases to CSV files.

        Stores weights in vendor directory as:
        - weights.csv: All weight matrix values (row_idx, col_idx)
        - biases.csv: All bias vector values

        CSV format is human-readable and easy to inspect or edit.

        Args:
            model: Trained PyTorch model
        """
        weights_data = []
        biases_data = []

        for name, param in model.named_parameters():
            if param.dim() == 2:  # Matrix weights (Linear layers)
                for row_idx in range(param.shape[0]):
                    for col_idx in range(param.shape[1]):
                        weights_data.append({
                            'layer_name': name,
                            'row_idx': row_idx,
                            'col_idx': col_idx,
                            'value': param[row_idx, col_idx].item()
                        })
            elif param.dim() == 1 and 'weight' in name:  # 1D weights (BatchNorm)
                for idx in range(param.shape[0]):
                    weights_data.append({
                        'layer_name': name,
                        'row_idx': idx,
                        'col_idx': 0,
                        'value': param[idx].item()
                    })
            elif param.dim() == 1 and 'bias' in name:  # Biases
                for idx in range(param.shape[0]):
                    biases_data.append({
                        'layer_name': name,
                        'bias_idx': idx,
                        'value': param[idx].item()
                    })

        weights_df = pd.DataFrame(weights_data)
        biases_df = pd.DataFrame(biases_data)

        weights_df.to_csv(self.vendor_dir / "weights.csv", index=False)
        biases_df.to_csv(self.vendor_dir / "biases.csv", index=False)

    def _load_weights(self, model: torch.nn.Module):
        """
        Load model weights and biases from CSV files.

        Args:
            model: PyTorch model to load weights into

        Returns:
            None (modifies model in-place)
        """
        weights_path = self.vendor_dir / "weights.csv"
        biases_path = self.vendor_dir / "biases.csv"

        if not weights_path.exists() or not biases_path.exists():
            return

        weights_df = pd.read_csv(weights_path)
        biases_df = pd.read_csv(biases_path)

        # Create lookup dicts for efficient access
        weight_dict = {}
        for _, row in weights_df.iterrows():
            layer = row['layer_name']
            if layer not in weight_dict:
                weight_dict[layer] = {}
            weight_dict[layer][(int(row['row_idx']), int(row['col_idx']))] = row['value']

        bias_dict = {}
        for _, row in biases_df.iterrows():
            layer = row['layer_name']
            if layer not in bias_dict:
                bias_dict[layer] = {}
            bias_dict[layer][int(row['bias_idx'])] = row['value']

        # Apply weights to model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() == 2 and name in weight_dict:
                    w = weight_dict[name]
                    shape = param.shape
                    tensor = torch.zeros(shape)
                    for (r, c), v in w.items():
                        if r < shape[0] and c < shape[1]:
                            tensor[r, c] = v
                    param.copy_(tensor)
                elif param.dim() == 1 and name in weight_dict:
                    w = weight_dict[name]
                    tensor = torch.zeros(param.shape)
                    for key, v in w.items():
                        idx = key[0] if isinstance(key, tuple) else key
                        if idx < param.shape[0]:
                            tensor[idx] = v
                    param.copy_(tensor)
                elif param.dim() == 1 and name in bias_dict:
                    b = bias_dict[name]
                    tensor = torch.zeros(param.shape)
                    for i, v in b.items():
                        if i < param.shape[0]:
                            tensor[i] = v
                    param.copy_(tensor)

    def get_model(self) -> VendorInvoiceModel:
        """
        Get model instance with loaded weights for inference.

        Returns:
            VendorInvoiceModel in eval mode with weights loaded
        """
        # Load vocabulary to get actual input dimension
        from ocr.text_processor import load_vocabulary
        vectorizer = load_vocabulary(self.vendor_id, self.vendor_dir)
        actual_input_dim = len(vectorizer.vocabulary_) if vectorizer else self.config['input_dim']

        label_encoders = self.metadata.get('label_encoders', {})
        num_inv = max(len(label_encoders.get('invoice_numbers', {})), 2)
        num_dates = max(len(label_encoders.get('invoice_dates', {})), 2)
        num_items = max(len(label_encoders.get('item_names', {})), 2)

        model = VendorInvoiceModel(
            input_dim=actual_input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout=0.0,  # No dropout during inference
            num_invoice_numbers=num_inv,
            num_dates=num_dates,
            num_items=num_items
        )

        self._load_weights(model)
        model.to(self.device)
        model.eval()
        return model
