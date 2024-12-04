import math
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict
from torchmetrics import Accuracy

def create_local_attention_mask(seq_len, n):
    """
    Creates a local attention mask allowing each token to attend only to
    tokens within +/- n positions.

    Args:
        seq_len (int): Length of the sequence.
        n (int): Number of tokens to the left and right a token can attend to.

    Returns:
        attention_mask (torch.Tensor): Attention mask of shape (seq_len, seq_len).
    """
    # Initialize a full attention mask (seq_len x seq_len)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    
    # Fill the mask to allow only local attention
    for i in range(seq_len):
        mask[i, max(0, i-n):min(seq_len, i+n+1)] = 0  # Allow +/- n tokens
    
    attention_mask = mask.float() * -1e9 # Make additive
    
    return attention_mask 

class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input embeddings, adding temporal information
    for sequence modeling tasks.

    Args:
        d_model (int): Dimension of the embedding space.
        dropout (float): Dropout rate applied to the positional encoding. Defaults to 0.1.
        max_len (int): Maximum length of sequences supported. Defaults to 5000.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape ``[batch_size, seq_len, embedding_dim]``.

        Returns:
            Tensor: Output tensor of shape ``[batch_size, seq_len, embedding_dim]``.
        """
        _, seq_len, _ = x.size()

        # Add positional encodings
        x = x + self.pe[:seq_len].unsqueeze(0)  # Shape: [1, seq_len, d_model]
        return self.dropout(x)


class ProsodyFeatureModel(nn.Module):
    """
    A model for extracting features from prosodic input using embeddings, positional
    encoding, and a Transformer encoder.

    Args:
        num_tokens (int): Number of unique tokens in the vocabulary.
        embedding_dim (int): Dimension of the embedding space. Defaults to 128.
        d_model (int): dimension for transformer model. Defaults to 512
        num_layers (int): Number of layers in the Transformer encoder. Defaults to 2.
        dropout (float): Dropout rate applied to embeddings and encoder. Defaults to 0.0.
        local_attn_mask (int): Size of models local attention field. Defaults to None (full sequence).
    """

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int = 128,
        d_model: int = 512,
        num_layers: int = 2,
        dropout: float = 0.0,
        local_attn_mask: int | None = None
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.local_attn_mask = local_attn_mask

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dim
        )

        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        
        self.linear = nn.Linear(in_features=embedding_dim, out_features=d_model)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dropout=dropout, batch_first=True
            ),
            num_layers=num_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes input tokens through embedding, positional encoding, and Transformer encoder.

        Args:
            x (Tensor): Input tensor of shape ``[batch_size, seq_len]``, where each entry
                        is a token index.

        Returns:
            Tensor: Encoded tensor of shape ``[batch_size, embedding_dim]`` representing
                    the mean-pooled sequence features.
        """
        # Embed tokens and apply positional encoding
        embeds = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        embeds_pe = self.pos_encoding(
            embeds
        )  # Shape: [batch_size, seq_len, embedding_dim]

        embeds_pe = self.linear(embeds_pe) # Shape: [batch_size, seq_len, d_model]
        
        if self.local_attn_mask:
            _, seq_len, _ = embeds_pe.shape
            mask = create_local_attention_mask(n=self.local_attn_mask, seq_len=seq_len).cuda()
        else:
            mask = None
            
        # Encode sequences using Transformer encoder
        z = self.encoder(embeds_pe, mask=mask)  # Shape: [batch_size, seq_len, embedding_dim]

        # Mean-pool along the sequence dimension
        z_mean = z.mean(dim=1)  # Shape: [batch_size, embedding_dim]
        return z_mean


class ProsodySpeakerVerificationModel(LightningModule):

    def __init__(
        self,
        num_speakers: int,
        hparams: dict = {},
        optimizer_params: dict = {},
        scheduler_params: dict = {}
    ) -> None:

        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        # Define loss and metric functions
        self.loss_fcn = nn.CrossEntropyLoss()
        self.metrics = torch.nn.ModuleDict(
            {"accuracy": Accuracy(task="multiclass", num_classes=num_speakers)}
        )

        # Define feature model and
        self.feature_model = ProsodyFeatureModel(**hparams)
        self.classifier = nn.Linear(
            in_features=hparams["d_model"], out_features=num_speakers
        )

    def configure_optimizers(self) -> Dict:
        """Configures optimizer

        Returns:
            optimizer_dict (Dict): configured optimizer and lr scheduler
        """
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = CosineAnnealingLR(optimizer=optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 'step' or 'epoch'
                "frequency": 1,       # Frequency of applying the scheduler
            },
        }

    def get_features(self, x: Tensor) -> Tensor:
        """Extracts hidden embeddings/features for sample x

        Args:
            x (Tensor): input

        Returns:
            z (Tensor): embeddings
        """
        with torch.no_grad():
            z = self.feature_model(x)

        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass function

        Args:
            x (Tensor): input

        Returns:
            y (Tensor): model output
        """

        z = self.feature_model(x)
        y = self.classifier(z)

        return y

    def training_step(self, batch: Any, batch_idx: int = 0) -> Any:
        """Performs training step with loss computation and metric logging

        Args:
            batch (Any): batch of samples (feats,labs)
            batch_idx (int, optional): Index of batch. Defaults to 0.

        Returns:
            loss (Any): batch loss
        """

        x, y_true = batch  # Unpack batch

        y_pred = self(x)  # Forward pass

        # Compute and log loss
        loss = self.loss_fcn(y_pred, y_true)
        self.log("train_loss", loss, sync_dist=True)

        # Compute and log metrics
        for metric_name, metric_fcn in self.metrics.items():
            metric_fcn = metric_fcn
            metric_val = metric_fcn(y_pred, y_true)
            self.log("train_%s" % metric_name, metric_val, sync_dist=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int = 0) -> Any:
        """Performs validation step with loss computation and metric logging

        Args:
            batch (Any): batch of samples
            batch_idx (int, optional): Index of batch. Defaults to 0.

        Returns:
            loss (Any): batch loss
        """

        x, y_true = batch  # Unpack batch

        y_pred = self(x)  # Forward pass

        # Compute and log loss
        loss = self.loss_fcn(y_pred, y_true)
        self.log("val_loss", loss, sync_dist=True)

        # Compute and log metrics
        for metric_name, metric_fcn in self.metrics.items():
            metric_val = metric_fcn(y_pred, y_true)
            self.log("val_%s" % metric_name, metric_val, sync_dist=True)

        return loss
