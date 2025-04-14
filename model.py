#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:11:19 2024

@author: federicodanesin

Description:
This script defines several layers and components for a deep learning model,
focusing on attention mechanisms and feature encoding.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    

def visualize_attention_scores(attn_scores, layer_name):
    """
    Visualizes attention scores from a given layer.
    Args:
        attn_scores (tf.Tensor): Attention scores with shape (batch, query, key).
        layer_name (str): Name of the layer generating the attention scores.
    """
    plt.figure(figsize=(10, 8))
    plt.title(f"Attention Scores from {layer_name}")
    plt.imshow(attn_scores[0], cmap='viridis', aspect='auto')  # For batch=1
    plt.colorbar()
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.show()


class TimeSeriesEmbedding(tf.keras.layers.Layer):
    """
    Applies dense feature projection and optional positional encoding.

    Args:
        d_model (int): Dimension of the output embedding.
        seq_len (int): Length of the input sequence.
        positional (bool): Whether to apply positional encoding.
    """
    def __init__(self, d_model, seq_len, positional=True):
        super(TimeSeriesEmbedding, self).__init__()
        self.positional = positional
        self.dense = tf.keras.layers.Dense(d_model, activation=None)
        self.positional_encoding = self.get_positional_encoding(seq_len, d_model)

    def get_positional_encoding(self, seq_len, d_model):
        """
        Generates sinusoidal positional encoding.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
        Returns:
            tf.Tensor: Positional encoding of shape (1, seq_len, d_model).
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        # Apply sine to even indices and cosine to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        """
        Applies dense projection and adds positional encoding if enabled.
        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, features).
        Returns:
            tf.Tensor: Embedded tensor.
        """
        x = self.dense(x)
        if self.positional:
            x += self.positional_encoding
        return x


class BaseAttention(tf.keras.layers.Layer):
    """
    Base class for attention layers, combining multi-head attention, residuals, and normalization.
    """
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()


class CrossAttention(BaseAttention):
    """
    Applies cross-attention between two tensors: query and context.
    """
    def call(self, x, context):
        """
        Args:
            x (tf.Tensor): Query tensor.
            context (tf.Tensor): Context tensor (key and value).
        Returns:
            tf.Tensor: Output tensor after attention.
            tf.Tensor: Attention scores.
        """
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        self.last_attn_scores = attn_scores  # Cache attention scores for visualization
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x, attn_scores


class GlobalSelfAttention(BaseAttention):
    """
    Applies self-attention to the input tensor.
    """
    def call(self, x, visualize=False):
        """
        Args:
            x (tf.Tensor): Input tensor (query, key, and value are the same).
            visualize (bool): If True, visualizes attention scores.
        Returns:
            tf.Tensor: Output tensor after attention.
            tf.Tensor: Attention scores.
        """
        attn_output, attn_scores = self.mha(query=x, value=x, key=x, return_attention_scores=True)
        self.last_attn_scores = attn_scores  # Cache attention scores

        if visualize:
            visualize_attention_scores(attn_scores, self.name)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x, attn_scores


class MultiScaleConvBlock(tf.keras.layers.Layer):
    """
    Multi-Scale Convolutional Block with attention-based feature fusion.
    """
    def __init__(self, output_dim, kernel_sizes=[3, 5, 7], dropout_rate=0.1):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = [
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(output_dim, k, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.ReLU()
            ]) for k in kernel_sizes
        ]

        # Feed-forward network for attention
        self.ffn1 = tf.keras.layers.Dense(output_dim // 4, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(len(kernel_sizes))
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        """
        Applies multi-scale convolutions and fuses the results with attention.
        Args:
            x (tf.Tensor): Input tensor of shape (batch, length, features).
        Returns:
            tf.Tensor: Processed tensor.
        """
        features = [conv(x) for conv in self.convs]
        stacked_features = tf.stack(features, axis=-1)
        # Attention weights
        attention_scores = self.ffn1(stacked_features)
        attention_scores = self.ffn2(attention_scores)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Controlliamo che la somma di ogni vettore di attenzione sia 1 (softmax property)
        sum_attention = tf.reduce_sum(attention_weights, axis=-1)  # Sommiamo lungo i kernel

        # Weighted feature fusion
        weighted_features = attention_weights * stacked_features
        fused_features = tf.reduce_sum(weighted_features, axis=-1)

        residual = self.add([x, fused_features])
        residual_norm = self.layernorm(residual)
        
        return residual_norm

class AttentionFusionBlock(tf.keras.layers.Layer):
    """
    Attention Fusion Block for combining vectors of different dimensions.
    """
    def __init__(self, output_dim=64, name=None):
        super(AttentionFusionBlock, self).__init__(name=name)

        # Projection layers for input vectors
        self.time_pooled_projection = tf.keras.layers.Dense(output_dim, activation='relu')
        self.channel_pooled_projection = tf.keras.layers.Dense(output_dim, activation='relu')
        # Attention calculation
        self.attention_dense = tf.keras.layers.Dense(1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, time_pooled, channel_pooled):
        """
        Combines input vectors using attention.
        Args:
            time_pooled(tf.Tensor): First input vector.
            channe_pooled (tf.Tensor): Second input vector.
        Returns:
            tf.Tensor: Fused vector.
        """
        time_pooled_projection = self.time_pooled_projection(time_pooled)
        channel_pooled_projection = self.channel_pooled_projection(channel_pooled)
        combined = tf.stack([time_pooled_projection, channel_pooled_projection], axis=1)
        attn_scores = self.attention_dense(combined)
        attn_weights = self.softmax(attn_scores)
        weighted_sum = tf.reduce_sum(attn_weights * combined, axis=1)
        return weighted_sum


class FeedForwardStack(tf.keras.layers.Layer):
    """
    Combines feed-forward and multi-scale convolutional blocks.
    """
    def __init__(self, num_layers, num_filters, attention_type='self', **kwargs):
        super().__init__()
        self.attention_type = attention_type
        self.layers = []

        for _ in range(num_layers):
            attention_block = GlobalSelfAttention(**kwargs) if attention_type == 'self' else CrossAttention(**kwargs)
            mscb = MultiScaleConvBlock(output_dim=num_filters, kernel_sizes=[1, 3, 7])

            self.layers.append({
                'attention': attention_block,
                'mscb': mscb
            })

    def call(self, x, context=None):
        """
        Passes input through all layers of attention and convolution.
        Args:
            x (tf.Tensor): Input tensor.
            context (tf.Tensor, optional): Context tensor for cross-attention.
        Returns:
            tf.Tensor: Processed tensor.
        """
        for layer in self.layers:
            attention_block = layer['attention']
            mscb = layer['mscb']
            attn_output, _ = attention_block(x) if self.attention_type == 'self' else attention_block(x, context)
            conv_output = mscb(attn_output)
        return x

class BaseEncoder(tf.keras.layers.Layer):
    """
    Base encoder class for applying positional encoding and attention-based feature extraction.
    """
    def __init__(self, num_layers, key_dim, num_filters, num_heads, seq_len, embedding_dim, positional_encoding=True, dropout_rate=0.1, name=None):
        super(BaseEncoder, self).__init__(name=name)
        self.pos_embedding = TimeSeriesEmbedding(d_model=embedding_dim, seq_len=seq_len, positional=positional_encoding)
        self.enc_layers = FeedForwardStack(num_layers=num_layers, num_filters=num_filters, num_heads=num_heads, key_dim=key_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()

    def apply_encoding(self, x):
        """
        Applies positional embedding, dropout, and attention-based feature extraction.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, features).
        Returns:
            tf.Tensor: Encoded tensor.
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.enc_layers(x)
        return x


class TimeEncoder(BaseEncoder):
    """
    Encoder specializing in temporal features.
    """
    def call(self, x):
        return self.apply_encoding(x)


class ChannelEncoder(BaseEncoder):
    """
    Encoder specializing in channel-wise features.
    """
    def __init__(self, num_layers, key_dim, num_filters, num_heads, seq_len, embedding_dim, positional_encoding=False, dropout_rate=0.5, name=None):
        super(ChannelEncoder, self).__init__(num_layers, key_dim, num_filters, num_heads, seq_len, embedding_dim, positional_encoding, dropout_rate, name=name)
        self.transpose_layer = tf.keras.layers.Permute((2, 1))

    def call(self, x):
        x = self.transpose_layer(x)
        return self.apply_encoding(x)

class MACTNet(tf.keras.Model):
    """
    Main model combining temporal and channel encoders with attention fusion.
    """
    def __init__(self, num_classes, use_time_encoder=True, use_channel_encoder=True, **kwargs):
        super(MACTNet, self).__init__()

        self.time_encoder_params = kwargs.pop("time_encoder_params", {})
        self.channel_encoder_params = kwargs.pop("channel_encoder_params", {})
        self.num_classes = num_classes
        self.use_time_encoder = use_time_encoder
        self.use_channel_encoder = use_channel_encoder

        self.time_encoder = TimeEncoder(**self.time_encoder_params, name="Time_Encoder") if use_time_encoder else None
        self.channel_encoder = ChannelEncoder(**self.channel_encoder_params, name="Channel_Encoder") if use_channel_encoder else None
        
        self.attention_fusion = AttentionFusionBlock(name="Attention_Fusion_Block")
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax', name="Output_Classifier")
        self.global_pool = tf.keras.layers.GlobalMaxPooling1D()

    def call(self, x, return_fusion_vectors=False):
        """
        Forward pass through the model.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, features).
            return_fusion_vectors (bool): If True, returns the fusion vectors instead of the classification output.

        Returns:
            tf.Tensor: Classification output or fusion vectors.
        """
        encoded_time = self.time_encoder(x) if self.use_time_encoder else None
        encoded_channel = self.channel_encoder(x) if self.use_channel_encoder else None

        if self.use_time_encoder:
            encoded_time = self.global_pool(encoded_time)
        if self.use_channel_encoder:
            encoded_channel = self.global_pool(encoded_channel)

        if self.use_time_encoder and self.use_channel_encoder:
            if return_fusion_vectors:
                fusion_vectors = self.attention_fusion(encoded_time, encoded_channel)
                return fusion_vectors
            else:
                fusion_output = self.attention_fusion(encoded_time, encoded_channel)
                output = self.classifier(fusion_output)
                return output
            
        elif self.use_time_encoder:
            return self.classifier(encoded_time) 
        elif self.use_channel_encoder:
            return self.classifier(encoded_channel)
        else:
            raise ValueError("At least one of 'use_time_encoder' or 'use_channel_encoder' must be True.")

    def predict_with_visualization(self, x):
        """
        Makes predictions and visualizes intermediate attention scores.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, features).

        Returns:
            tf.Tensor: Classification output.
        """
        encoded_time = self.time_encoder(x)
        encoded_channel = self.channel_encoder(x)

        visualize_attention_scores(self.time_encoder.enc_layers.layers[0]['attention'].last_attn_scores, "Time Encoder Attention")
        visualize_attention_scores(self.channel_encoder.enc_layers.layers[0]['attention'].last_attn_scores, "Channel Encoder Attention")

        encoded_time = self.global_pool(encoded_time)
        encoded_channel = self.global_pool(encoded_channel)
        feature_fusion = self.attention_fusion(encoded_time, encoded_channel)
        output = self.classifier(feature_fusion)

        return output
    
    def get_config(self):
        """
        Returns the configuration of the model for serialization.
        """
        config = super(MACTNet, self).get_config()
        config.update({
            "num_classes": self.num_classes,
            "use_time_encoder": self.use_time_encoder,
            "use_channel_encoder": self.use_channel_encoder,
            "time_encoder_params": self.time_encoder_params,
            "channel_encoder_params": self.channel_encoder_params
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates a model instance from its config.
        """
        return cls(
            num_classes=config["num_classes"],
            use_time_encoder=config["use_time_encoder"],
            use_channel_encoder=config["use_channel_encoder"],
            time_encoder_params=config["time_encoder_params"],
            channel_encoder_params=config["channel_encoder_params"]
    )
    
    def evaluate_model(self, X_val, y_val, activity_labels):
        """
        Evaluate the model and print metrics, including inference time.
    
        Args:
            model (tensorflow.keras.Model): Trained model.
            X_val (numpy.ndarray): Validation data (features).
            y_val (numpy.ndarray): True labels for validation data (one-hot encoded).
            activity_labels (list): List of activity labels for the classification task.
        """

        print("Starting inference...")
        start_time = time.time()
        predictions = self.predict(X_val, batch_size=32)
        inference_time = time.time() - start_time
    
        print(f"Total inference time: {inference_time:.2f}s")
        print(f"Average time per sample: {inference_time / len(X_val):.6f}s")
    
        # Convert predictions and true labels to class indices
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = tf.argmax(y_val, axis=1).numpy()
    
        # Identify classes present in the dataset
        present_classes = np.unique(true_classes)
        adjusted_labels = [activity_labels[i] for i in present_classes]
    
        # Print classification metrics
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, labels=present_classes, target_names=adjusted_labels))
    
        # Compute and display the confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_classes, predicted_classes, labels=present_classes, normalize='true')
        print(cm)
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=adjusted_labels, yticklabels=adjusted_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Normalized Confusion Matrix')
        plt.show()


class TrainingPerformanceCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f}s")

        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        print(f"Memory usage: {memory_usage:.2f} MB")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        print(f"Total training time: {total_time:.2f}s")
        print(f"Average time per epoch: {sum(self.epoch_times) / len(self.epoch_times):.2f}s")


if __name__ == "__main__":
    N = 32
    seq_len = 300
    num_features = 128
    num_layers = 2
    num_classes = 8
    num_heads = 4
    key_dim = 128
    embedding_dim = 128

    x = tf.random.normal((N, seq_len, num_features))

    encoder_param = {
        "num_layers": num_layers,
        "num_filters": embedding_dim,
        "key_dim": key_dim,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "embedding_dim": embedding_dim
    }

    caf = MACTNet(num_classes, time_encoder_params=encoder_param, channel_encoder_params=encoder_param)
    output = caf(x)
    
    a = MultiScaleConvBlock(embedding_dim)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    caf.summary()


    