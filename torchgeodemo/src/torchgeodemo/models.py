# Libraries
from typing import Literal
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import copy

# This project's modules
from .activation import TopK
from .loss import normalized_mean_squared_error, normalized_L1_loss


class MLP(pl.LightningModule):
    """
    Multi-Layer Perceptron (MLP) model implemented using PyTorch Lightning.

    Args:
        size_sequence (list[int]): A list of integers specifying the size of each layer in the MLP.
        final_activation (Literal["Identity", "ReLU", "Tanh", "Sigmoid"], optional): The activation function to use for the final layer. Defaults to "Identity".
        final_activation_topk (bool, optional): Whether to apply a TopK activation function to the final layer. Defaults to False.
        final_activation_topk_k (int, optional): The number of top elements to keep if using TopK activation. If None, defaults to half of the output size. Defaults to None.
        negative_slope (float, optional): The negative slope value for the LeakyReLU activation function. Defaults to 0.01.

    Attributes:
        size_sequence (list[int]): The size of each layer in the MLP.
        final_activation (str): The activation function for the final layer.
        final_activation_topk (bool): Whether a TopK activation function has been to the final layer.
        final_activation_topk_k (int): The number of top elements kept if using TopK activation.
        mlp (torch.nn.Sequential): The sequential container holding the MLP layers.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            MLP forward pass. Takes an input tensor `x` and returns the output tensor after passing through the MLP.
    """
    def __init__(self, 
            size_sequence: list[int], 
            final_activation: Literal["Identity", "ReLU", "Tanh", "Sigmoid"] = "Identity",
            final_activation_topk: bool = False,
            final_activation_topk_k: int = None,
            negative_slope=0.01
            ) -> None:
        super(MLP, self).__init__()
        # Set parameters
        self.size_sequence = size_sequence
        self.final_activation = final_activation
        self.final_activation_topk = final_activation_topk
        self.final_activation_topk_k = final_activation_topk_k
        # Create Multi-Layer Perceptron based on size sequence
        self.mlp = torch.nn.Sequential()
        # Add all layers but the last
        for i in range(len(self.size_sequence) - 2):
            self.mlp.append(
                torch.nn.Linear(self.size_sequence[i], self.size_sequence[i + 1]))
            self.mlp.append(
                torch.nn.LeakyReLU(negative_slope=negative_slope))
        # Add final layer
        self.mlp.append(
            torch.nn.Linear(self.size_sequence[-2], self.size_sequence[-1]))
        # Add final activation
        # If specified, add TopK activation
        if self.final_activation_topk:
            if self.final_activation_topk_k is None:
                # If not specified, use half of the output size
                self.final_activation_topk_k = math.floor(self.size_sequence[-1] / 2)
            self.mlp.append(
                TopK(self.final_activation_topk_k, self.final_activation))
        # Otherwise, add standard activation
        else:
            if self.final_activation == 'ReLU':
                self.mlp.append(nn.ReLU())
            elif self.final_activation == 'Tanh':
                self.mlp.append(nn.Tanh())
            elif self.final_activation == 'Sigmoid':
                self.mlp.append(nn.Sigmoid())
            else:
                self.mlp.append(nn.Identity())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AutoEncoder(pl.LightningModule):
    """
    AutoEncoder implemented in PyTorch Lightning.

    Args:
        encoder_sizes (list[int]): List of integers specifying the sizes of the encoder layers.
        encoder_sparse (bool, optional): If True, creates a sparse encoder. Default is False.
        encoder_sparse_topk_k (int, optional): If encoder_sparse is True, specifies the top-k sparsity constraint. Default is None.
        encoder_activation (Literal["Identity", "ReLU", "Tanh", "Sigmoid"], optional): Final activation function for the encoder. Default is "Identity".
        decoder_sizes (list[int], optional): List of integers specifying the sizes of the decoder layers. If None, the decoder sizes are set to be the reverse of encoder sizes. Default is None.
        decoder_activation (Literal["Identity", "ReLU", "Tanh", "Sigmoid"], optional): Final activation function for the decoder. Default is "Identity".
        sparsity_loss_weight (float, optional): Weight for the sparsity loss if a sparse autoencoder is created. Default is 0.01.
        use_covariance_loss (bool, optional): If True, includes covariance loss in training. Default is True.
        verbose (bool, optional): If True, prints additional information. Default is False.

    Methods:
        forward(x):
            Forward pass through the autoencoder. Returns embeddings and reconstruction.
        
        encode(x):
            Encodes the input data and returns the embeddings.
        
        training_step(batch, batch_idx):
            Defines the training step. Computes the loss and logs it.
        
        configure_optimizers():
            Configures the optimizer and learning rate scheduler.
    """
    def __init__(self, 
            encoder_sizes: list[int], 
            encoder_sparse: bool = False,
            encoder_sparse_topk_k: int = None,
            encoder_activation: Literal["Identity", "ReLU", "Tanh", "Sigmoid"] = "Identity",
            decoder_sizes: list[int] = None, 
            decoder_activation: Literal["Identity", "ReLU", "Tanh", "Sigmoid"] = "Identity", 
            sparsity_loss_weight: float = 0.01,
            use_covariance_loss: bool = True,
            dcc_from_epoch = float('inf'),
            dcc_neighbours = 1,
            dcc_nn_lambda = 1.0,
            verbose: bool = False
            ) -> None:
        super(AutoEncoder, self).__init__()

        # Set parameters
        self.encoder_sizes = encoder_sizes
        self.encoder_sparse = encoder_sparse
        self.encoder_sparse_topk_k = encoder_sparse_topk_k
        self.encoder_activation = encoder_activation
        self.decoder_sizes = decoder_sizes
        self.decoder_activation = decoder_activation
        self.sparsity_loss_weight = sparsity_loss_weight
        self.use_covariance_loss = use_covariance_loss

        # Deep Continuous Clustering (DCC)
        self.dcc_from_epoch = dcc_from_epoch
        self.dcc_neighbours = dcc_neighbours
        self.dcc_nn_lambda = dcc_nn_lambda
        self.dcc_train = False
        self.dcc_pretrained_encoder = None
        self.dcc_criterion = nn.MSELoss()
        # print(f"{self.dcc_from_epoch=}", flush=True)
        
        # Encoder
        self.encoder = MLP(
            self.encoder_sizes, 
            self.encoder_activation, 
            self.encoder_sparse, 
            self.encoder_sparse_topk_k)
        # Decoder
        if decoder_sizes is None:
            # If not specified, assume symmetrical autoencoder and use reverse of encoder sizes
            self.decoder_sizes = list(reversed(encoder_sizes))
            print(f"Decoder sizes not specified. Using reverse of encoder sizes: {self.decoder_sizes}.") if verbose else None
        self.decoder = MLP(
            self.decoder_sizes, 
            self.decoder_activation)
        
    # Forward pass
    def forward(self, x):
        # Encode and decode
        embeddings = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        # Return embeddings and reconstruction
        return embeddings, reconstruction
    # Encoding only
    def encode(self, x):
        # Return encodeer output
        return self.encoder(x)
    
    # Set up DCC training
    def on_train_epoch_end(self):
        # At self.dcc_from_epoch epochs
        # if the model was not being trained in DCC mode
        # but it is going to be trained in DCC mode from now on
        # thus, save the encoder to generate the pretrained_embeddings
        if self.current_epoch == self.dcc_from_epoch:
            if not self.dcc_train:
                self.dcc_pretrained_encoder = copy.deepcopy(self.encoder)
                self.dcc_pretrained_encoder.eval()
                # print("Starting DCC training. Pretrained encoder saved.", flush=True)
        self.dcc_train = self.current_epoch >= self.dcc_from_epoch
        # print(f"{self.current_epoch} - {self.dcc_train=}", flush=True)
    
    # Training step
    def training_step(self, batch, batch_idx):
        # Forward pass
        embeddings, reconstruction = self.forward(batch)

        # Reconstruction loss
        loss = normalized_mean_squared_error(reconstruction, batch)
        self.log('recon_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Covariance loss (optional)
        if self.use_covariance_loss:
            # Create centered embeddings
            batch_size = embeddings.size(0)
            embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
            # Calculate covariance matrix
            cov_matrix = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
            # Sum of squares of elements, excluding the diagonal
            covariance_loss = (cov_matrix * (
                    1 - torch.eye(cov_matrix.size(0), device=embeddings.device))
                ).abs().sum()
            self.log('covariance_loss', covariance_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss += 1.0 * covariance_loss

        # Sparsity loss
        if self.encoder_sparse:
            sparsity_loss = normalized_L1_loss(embeddings, batch)
            self.log('sparsity_loss', sparsity_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss += self.sparsity_loss_weight * sparsity_loss
    
        # DCC training
        if self.dcc_train:
            # MSE loss between pretrained and current embeddings
            with torch.no_grad():
                pretrained_embeddings = self.dcc_pretrained_encoder(batch)
                pretrained_embeddings_loss = self.dcc_criterion(pretrained_embeddings, embeddings)
            # Average distance of neighbours
            # get k nearest neighbours (+1 as self is included)
            #
            # torch.topk
            # https://pytorch.org/docs/stable/generated/torch.topk.html
            # Returns the k largest elements of the given input tensor along a given dimension.
            # If largest is False then the k smallest elements are returned.
            nearest_neighbours, _ = torch.topk(
                # torch.cdist
                # https://pytorch.org/docs/stable/generated/torch.cdist.html
                # Computes batched the p-norm distance between each pair of the two collections of row vectors.
                torch.cdist(embeddings, embeddings, p=2), 
                k=self.dcc_neighbours+1, 
                largest=False
                )
            # remove self distance
            nearest_neighbours = nearest_neighbours[:, 1:]
            # calcualte mean and standard deviation 
            nn_std, nn_mean = torch.std_mean(nearest_neighbours, dim=0)
            # mask values, excluding those higher than (mean - std)
            nearest_neighbours_mask = (nearest_neighbours < (nn_mean - nn_std)).int()
            nearest_neighbours = nearest_neighbours * nearest_neighbours_mask
            #
            # Additional estimators could be added above, as per the DCC paper, e.g.
            # reps_dist = (mu_1 * reps_dist**2) / (mu_1 + reps_dist**2)
            # mu_1 = 1.0
            # mu_2 = 1.0
            # zknn_dist = (mu_2 * zknn_dist**2) / (mu_2 + zknn_dist**2)
            #
            # Overall DCC loss
            if torch.sum(nearest_neighbours_mask) == 0:
                # if no masked-in values, i.e., no neighbours within threshold
                # only use pretrained embeddings loss
                dcc_loss = pretrained_embeddings_loss
            else:
                # otherwise, use pretrained embeddings loss and
                # average of squares, counting only masked-in values
                dcc_loss = pretrained_embeddings_loss + self.dcc_nn_lambda * (
                    torch.sum(torch.square(nearest_neighbours)) / torch.sum(nearest_neighbours_mask))
            # Log losses
            self.log('pretrained_embeddings_loss', pretrained_embeddings_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('nearest_neighbours_loss', (torch.sum(torch.square(nearest_neighbours)) / torch.sum(nearest_neighbours_mask)), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('dcc_loss', dcc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss += dcc_loss
    
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Return loss
        return loss
    
    # Optimizer details
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss_epoch'}
