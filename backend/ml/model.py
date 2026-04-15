"""
PyTorch Neural Network Model for Invoice Field Extraction.

A feedforward neural network that predicts invoice fields from TF-IDF features.
Uses separate classification heads for categorical fields (invoice number, date, item name)
and regression heads for continuous values (amounts, quantities).

Architecture:
- Input: TF-IDF feature vector (configurable size, default 500 dims)
- Hidden layers: configurable [128, 64, 32] with BatchNorm and Dropout
- Output heads: 3 classification + 4 regression = 7 total fields

The model is vendor-specific - each vendor has their own trained model.
"""

import torch
import torch.nn as nn
import numpy as np


class VendorInvoiceModel(nn.Module):
    """
    Feedforward neural network for invoice field extraction.

    Input: TF-IDF feature vector (default 500 dims)
    Output: 7 fields - 3 classification (invoice_number, date, item_name)
            + 4 regression (invoice_amount, quantity, unit_price, total_price)

    The model uses:
    - Shared feature extractor (hidden layers)
    - Separate output heads for each field type
    - Softmax for classification outputs (probabilities)
    - Raw values for regression outputs
    """

    def __init__(self, input_dim=500, hidden_dims=[128, 64, 32], dropout=0.3,
                 num_invoice_numbers=100, num_dates=50, num_items=200):
        """
        Initialize the neural network with specified architecture.

        Args:
            input_dim: Size of input TF-IDF feature vector
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability for regularization
            num_invoice_numbers: Number of unique invoice number classes
            num_dates: Number of unique date format classes
            num_items: Number of unique item name classes
        """
        super().__init__()

        # -------------------------------------------------------------------------
        # Shared feature extractor - learns common patterns across all fields
        # Uses BatchNorm for training stability and Dropout for regularization
        # -------------------------------------------------------------------------
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # -------------------------------------------------------------------------
        # Classification heads - output class probabilities (softmax)
        # Used for categorical fields where we predict one of N classes
        # -------------------------------------------------------------------------
        self.invoice_number_head = nn.Linear(prev_dim, num_invoice_numbers)
        self.invoice_date_head = nn.Linear(prev_dim, num_dates)
        self.item_name_head = nn.Linear(prev_dim, num_items)

        # -------------------------------------------------------------------------
        # Regression heads - output continuous values
        # Used for numeric fields where any value is possible
        # -------------------------------------------------------------------------
        self.invoice_amount_head = nn.Linear(prev_dim, 1)
        self.item_quantity_head = nn.Linear(prev_dim, 1)
        self.per_item_price_head = nn.Linear(prev_dim, 1)
        self.total_item_price_head = nn.Linear(prev_dim, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict with all output tensors
        """
        features = self.feature_extractor(x)

        return {
            'invoice_number': self.softmax(self.invoice_number_head(features)),
            'invoice_date': self.softmax(self.invoice_date_head(features)),
            'item_name': self.softmax(self.item_name_head(features)),
            'invoice_amount': self.invoice_amount_head(features),
            'item_quantity': self.item_quantity_head(features),
            'per_item_price': self.per_item_price_head(features),
            'total_item_price': self.total_item_price_head(features)
        }

    def predict(self, x):
        """
        Inference mode - returns argmax for classification, raw for regression.

        Uses no-gradient mode for efficient inference. For classification,
        returns the index of the most likely class.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predictions dict, confidences dict)
        """
        with torch.no_grad():
            outputs = self.forward(x)

        # Get predictions - argmax for classification, raw value for regression
        predictions = {
            'invoice_number': torch.argmax(outputs['invoice_number'], dim=1).item(),
            'invoice_date': torch.argmax(outputs['invoice_date'], dim=1).item(),
            'item_name': torch.argmax(outputs['item_name'], dim=1).item(),
            'invoice_amount': outputs['invoice_amount'].item(),
            'item_quantity': outputs['item_quantity'].item(),
            'per_item_price': outputs['per_item_price'].item(),
            'total_item_price': outputs['total_item_price'].item()
        }

        # Confidence is max probability for classification, 1.0 for regression
        confidences = {
            'invoice_number': outputs['invoice_number'].max(dim=1).values.item(),
            'invoice_date': outputs['invoice_date'].max(dim=1).values.item(),
            'item_name': outputs['item_name'].max(dim=1).values.item(),
            'invoice_amount': 1.0,  # Regression doesn't have confidence
            'item_quantity': 1.0,
            'per_item_price': 1.0,
            'total_item_price': 1.0
        }

        return predictions, confidences


def get_model_config():
    """
    Return default model configuration.

    Returns:
        dict: Default hyperparameters for model architecture and training
    """
    return {
        'input_dim': 500,
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'num_invoice_numbers': 100,
        'num_dates': 50,
        'num_items': 200
    }
