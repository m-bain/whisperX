import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict
from torchmetrics import Accuracy
from transformers import WavLMForXVector

class SpeakerRecogModel(LightningModule):

    def __init__(
        self,
        model_name: str,
        num_speakers: int,
        freeze_feature_extractor: bool = False,
        optimizer_params: dict = {},
        scheduler_params: dict = {}
    ) -> None:

        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.model_name = model_name
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.freeze_feature_extractor = freeze_feature_extractor

        # Define loss and metric functions
        self.loss_fcn = nn.CrossEntropyLoss()
        self.metrics = torch.nn.ModuleDict(
            {"accuracy": Accuracy(task="multiclass", num_classes=num_speakers)}
        )

        # Define model and featurizer
        if model_name == 'wavlm':
            self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
            embed_dim = 512
        elif model_name == 'speechbrain':
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                                        savedir="/home1/nmehlman/models/spkrec-xvect-voxceleb")
            embed_dim = 512
        else:
            raise ValueError("Model name not recognized")
        
        # Define linear classifier
        self.classifer = nn.Linear(embed_dim, num_speakers)

    def configure_optimizers(self) -> Dict:
        """Configures optimizer

        Returns:
            optimizer_dict (Dict): configured optimizer and lr scheduler
        """

        if self.freeze_feature_extractor:
            params = self.classifer.parameters()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            params = self.parameters()

        optimizer = Adam(params, **self.optimizer_params)
        scheduler = CosineAnnealingLR(optimizer=optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 'step' or 'epoch'
                "frequency": 1,       # Frequency of applying the scheduler
            },
        }
        
    def _forward(self, x: Tensor) -> Tensor:
        if self.model_name == 'wavlm':
            return self.model(x).embeddings
        elif self.model_name == 'speechbrain':
            return self.model.encode_batch(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass function

        Args:
            x (Tensor): input

        Returns:
            y (Tensor): model output
        """
        
        if self.freeze_feature_extractor: # Get embeddings
            with torch.no_grad():
                embeddings = self._forward(x)
        else:
            embeddings = self._forward(x)
        
        y = self.classifer(embeddings) # Linear classifier

        return y
    
    def get_embeddings(self, x: Tensor) -> Tensor:
        """Get embeddings from the model

        Args:
            x (Tensor): input

        Returns:
            embeddings (Tensor): model embeddings
        """
        
        with torch.no_grad():
            embeddings = self._forward(x)
            
        return embeddings

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
