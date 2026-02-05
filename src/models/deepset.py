"""
DeepSet Model Architecture

This module implements the DeepSet neural network for superconductor
critical temperature prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DeepSetSuperconductor(tf.keras.Model):
    """
    DeepSet neural network for superconductor Tc prediction.
    
    Architecture:
    1. φ-network: Per-element feature extraction (permutation-invariant encoder)
    2. Sum pooling: Aggregate element representations (permutation-invariant operation)
    3. ρ-network: Process aggregated representation to predict Tc
    
    Reference:
        Zaheer et al., "Deep Sets", NeurIPS 2017
        https://arxiv.org/abs/1703.06114
    """
    
    def __init__(
        self,
        latent_dim: int = 300,
        phi_layers: Optional[List[int]] = None,
        rho_layers: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize DeepSet model.
        
        Args:
            latent_dim: Dimension of latent space after φ-network
            phi_layers: List of layer sizes for φ-network (per-element encoder)
            rho_layers: List of layer sizes for ρ-network (aggregator)
            **kwargs: Additional arguments for tf.keras.Model
        """
        super(DeepSetSuperconductor, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        
        # Default φ-network architecture
        if phi_layers is None:
            phi_layers = [992, 768, 512, 384, 256, 128, latent_dim]
        else:
            # Ensure last layer matches latent_dim
            if phi_layers[-1] != latent_dim:
                phi_layers = phi_layers[:-1] + [latent_dim]
        
        # Default ρ-network architecture
        if rho_layers is None:
            rho_layers = [960, 832, 768, 640, 512, 384, 256, 192, 160, 128, 96, 64, 1]
        
        self.phi_layer_sizes = phi_layers
        self.rho_layer_sizes = rho_layers
        
        # Build φ-network (per-element feature extractor)
        phi_network_layers = []
        for i, units in enumerate(phi_layers[:-1]):
            phi_network_layers.append(
                layers.Dense(units, activation='relu', name=f'phi_{i+1}')
            )
        # Last layer is linear
        phi_network_layers.append(
            layers.Dense(phi_layers[-1], activation='linear', name=f'phi_{len(phi_layers)}')
        )
        
        self.phi_network = tf.keras.Sequential(phi_network_layers, name='phi_network')
        
        # Build ρ-network (aggregator over pooled representation)
        rho_network_layers = []
        for i, units in enumerate(rho_layers[:-1]):
            rho_network_layers.append(
                layers.Dense(units, activation='relu', name=f'rho_{i+1}')
            )
        # Last layer is linear (regression output)
        rho_network_layers.append(
            layers.Dense(rho_layers[-1], activation='linear', name='rho_output')
        )
        
        self.rho_network = tf.keras.Sequential(rho_network_layers, name='rho_network')
        
        logger.info(f"DeepSet model initialized:")
        logger.info(f"  φ-network layers: {phi_layers}")
        logger.info(f"  ρ-network layers: {rho_layers}")
        logger.info(f"  Latent dimension: {latent_dim}")
    
    def call(self, inputs, training=None):
        """
        Forward pass of DeepSet model.
        
        Args:
            inputs: Tensor of shape (batch_size, max_elements, feature_dim)
            training: Boolean or None, whether in training mode
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size = tf.shape(inputs)[0]
        max_elements = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        # Flatten across elements to apply φ element-wise
        # Shape: (batch_size * max_elements, feature_dim)
        reshaped_inputs = tf.reshape(inputs, [batch_size * max_elements, feature_dim])
        
        # φ-network: Extract features per element
        # Shape: (batch_size * max_elements, latent_dim)
        phi_outputs = self.phi_network(reshaped_inputs, training=training)
        
        # Reshape back to (batch_size, max_elements, latent_dim)
        phi_outputs = tf.reshape(phi_outputs, [batch_size, max_elements, self.latent_dim])
        
        # Sum pooling across elements (permutation-invariant aggregation)
        # Shape: (batch_size, latent_dim)
        pooled = tf.reduce_sum(phi_outputs, axis=1)
        
        # ρ-network: Process aggregated representation
        # Shape: (batch_size, 1)
        output = self.rho_network(pooled, training=training)
        
        return output
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'phi_layers': self.phi_layer_sizes,
            'rho_layers': self.rho_layer_sizes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
    
    def summary(self, **kwargs):
        """Print model summary."""
        logger.info("φ-network (per-element encoder):")
        self.phi_network.summary(**kwargs)
        logger.info("\nρ-network (aggregator):")
        self.rho_network.summary(**kwargs)


def create_deepset_model(config: dict) -> DeepSetSuperconductor:
    """
    Create DeepSet model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DeepSet model
    """
    model_config = config.get('model', {})
    features_config = config.get('features', {})
    
    latent_dim = model_config.get('latent_dim', 300)
    phi_layers = model_config.get('phi_layers')
    rho_layers = model_config.get('rho_layers')
    max_elements = features_config.get('max_elements', 10)
    feature_dim = features_config.get('feature_dim', 23)
    
    # Create model
    model = DeepSetSuperconductor(
        latent_dim=latent_dim,
        phi_layers=phi_layers,
        rho_layers=rho_layers
    )
    
    # Build model
    model.build(input_shape=(None, max_elements, feature_dim))
    
    logger.info(f"DeepSet model created with input shape: (None, {max_elements}, {feature_dim})")
    
    return model


def compile_model(
    model: DeepSetSuperconductor,
    config: dict
) -> DeepSetSuperconductor:
    """
    Compile DeepSet model with optimizer and loss.
    
    Args:
        model: DeepSet model instance
        config: Configuration dictionary
        
    Returns:
        Compiled model
    """
    training_config = config.get('training', {})
    
    learning_rate = training_config.get('learning_rate', 0.001)
    optimizer_name = training_config.get('optimizer', 'adam')
    loss_name = training_config.get('loss', 'mse')
    
    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_name,
        metrics=['mae', 'mse']
    )
    
    logger.info(f"Model compiled with optimizer={optimizer_name}, lr={learning_rate}, loss={loss_name}")
    
    return model
