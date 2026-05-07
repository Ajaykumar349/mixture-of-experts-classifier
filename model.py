"""
Mixture of Experts (MoE) Classifier using TensorFlow
Author: Ajay Kumar Gangwar | NIT Durgapur | MTech AI & DS
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class Expert(tf.keras.Model):
    """Single Expert Network - specialized MLP"""
    
    def __init__(self, hidden_dim=128, output_dim=10, dropout_rate=0.2):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.output_layer = layers.Dense(output_dim)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)


class SparseRouter(layers.Layer):
    """Router that decides which experts to use for each input"""
    
    def __init__(self, num_experts, top_k=2, noise_std=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.router_layer = None  # Will be built in build()
        
    def build(self, input_shape):
        self.router_layer = layers.Dense(self.num_experts)
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        # Get routing logits
        router_logits = self.router_layer(inputs)
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = tf.random.normal(
                shape=tf.shape(router_logits), 
                stddev=self.noise_std
            )
            router_logits = router_logits + noise
        
        # Get routing probabilities
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = tf.math.top_k(router_probs, k=self.top_k)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / tf.reduce_sum(top_k_probs, axis=-1, keepdims=True)
        
        return top_k_probs, top_k_indices, router_probs


class MixtureOfExperts(Model):
    """
    Mixture of Experts Model
    - num_experts: Number of expert networks
    - top_k: Number of experts to use per input
    - load_balancing_weight: Weight for load balancing loss
    """
    
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=10, 
                 num_experts=4, top_k=2, load_balancing_weight=0.01):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight
        
        # Input projection layer (optional - if input_dim doesn't match data)
        self.input_proj = layers.Dense(input_dim)
        self.input_dim = input_dim
        
        # Create multiple expert networks
        self.experts = [
            Expert(hidden_dim, output_dim) 
            for _ in range(num_experts)
        ]
        
        # Router decides which experts to use
        self.router = SparseRouter(num_experts, top_k)
        
        # Track router loss
        self.router_loss_tracker = tf.keras.metrics.Mean(name="router_loss")
        
    def call(self, inputs, training=False):
        # Project inputs if needed
        if inputs.shape[-1] != self.input_dim:
            inputs = self.input_proj(inputs)
            
        batch_size = tf.shape(inputs)[0]
        
        # Get routing decisions
        routing_weights, expert_indices, router_probs = self.router(inputs, training)
        
        # Initialize final output
        final_output = tf.zeros((batch_size, 10))
        
        # For load balancing calculation
        expert_usage = tf.zeros(self.num_experts)
        router_prob_sum = tf.zeros(self.num_experts)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find which samples this expert should process
            mask = tf.reduce_any(expert_indices == expert_idx, axis=1)
            
            if tf.reduce_any(mask):
                # Get expert's output for assigned inputs
                expert_input = tf.boolean_mask(inputs, mask)
                expert_output = expert(expert_input, training=training)
                
                # Get corresponding routing weights
                batch_indices = tf.where(mask)[:, 0]
                router_weights_for_batch = tf.gather(routing_weights, batch_indices)
                
                # Find weight for this expert
                weight_mask = tf.equal(
                    tf.gather(expert_indices, batch_indices), 
                    expert_idx
                )
                expert_weights = tf.boolean_mask(router_weights_for_batch, weight_mask)
                expert_weights = tf.expand_dims(expert_weights, axis=1)
                
                # Add weighted contribution
                weighted_output = expert_output * expert_weights
                
                # Scatter back to original batch order
                indices = tf.cast(batch_indices, tf.int32)
                final_output = tf.tensor_scatter_nd_add(
                    final_output, 
                    tf.expand_dims(indices, axis=1), 
                    weighted_output
                )
                
                # Track usage for load balancing
                expert_usage = tf.tensor_scatter_nd_add(
                    expert_usage,
                    [[expert_idx]],
                    [tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32)), tf.float32)]
                )
                
                # Track router probability sum
                router_prob_for_expert = tf.reduce_sum(
                    tf.boolean_mask(router_probs[:, expert_idx], mask)
                )
                router_prob_sum = tf.tensor_scatter_nd_add(
                    router_prob_sum,
                    [[expert_idx]],
                    [router_prob_for_expert]
                )
        
        # Calculate load balancing loss
        total_samples = tf.cast(batch_size, tf.float32)
        expert_usage_prob = expert_usage / total_samples
        router_prob_avg = router_prob_sum / total_samples
        
        # Importance loss: coefficient of variation squared
        importance_loss = tf.reduce_mean(
            expert_usage_prob * router_prob_avg
        ) * self.num_experts
        
        # Store loss for tracking
        self.router_loss = importance_loss * self.load_balancing_weight
        self.router_loss_tracker.update_state(self.router_loss)
        
        return final_output
    
    @property
    def metrics(self):
        return [self.router_loss_tracker]
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # classification loss
            classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
            classification_loss = tf.reduce_mean(classification_loss)
            # Total loss
            total_loss = classification_loss + self.router_loss
            
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        return {
            "loss": total_loss,
            "classification_loss": classification_loss,
            "router_loss": self.router_loss,
            **{m.name: m.result() for m in self.metrics}
        }
