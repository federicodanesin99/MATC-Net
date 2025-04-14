#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:00:50 2025

@author: federicodanesin

Description:
This script implements testing and transfer learning for a deep learning model (MACTNet).
It includes attention fusion vector extraction, dataset sampling, PCA visualization,
and transfer learning functionality.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from preprocessing import preprocess_datasets, create_tf_dataset, create_label_encoder, print_class_distribution
from model import MACTNet, TrainingPerformanceCallback
from config import (
    batch_size,
    sequence_length,
    overlap,
    ant_drop,
    activity_labels,
    time_encoder_params,
    channel_encoder_params,
    num_classes,
    num_features,
    print_parameters,
    retrain_epochs,
    test_dataset, 
    retrain_rate,
    label_mapping
)

# Function to extract attention fusion vectors
def get_attention_fusion_vectors(model, dataset):
    """
    Extracts attention fusion vectors from the model for all samples in the dataset.

    Args:
        model (tf.keras.Model): Trained MACTNet model.
        dataset (tf.data.Dataset): Dataset with input data and labels.

    Returns:
        np.array: Fusion vectors.
        np.array: Corresponding labels.
    """
    all_fusion_vectors = []
    all_labels = []

    for batch_data, batch_labels in dataset:
        # Get attention fusion vectors
        fusion_vectors = model(batch_data, return_fusion_vectors=True)
        all_fusion_vectors.append(fusion_vectors.numpy())
        all_labels.append(batch_labels.numpy())

    return np.vstack(all_fusion_vectors), np.vstack(all_labels)

# Function to sample N examples per label
def sample_n_per_label(data, labels, n):
    """
    Samples N examples for each label from a dataset with one-hot encoded labels.

    Args:
        data (np.array): Dataset features.
        labels (np.array): One-hot encoded labels.
        n (int): Number of samples to extract per label.

    Returns:
        np.array: Sampled features.
        np.array: Corresponding one-hot encoded labels.
    """
    sparse_labels = np.argmax(labels, axis=1)
    unique_labels = np.unique(sparse_labels)
    sampled_data = []
    sampled_labels = []

    for label in unique_labels:
        indices = np.where(sparse_labels == label)[0]
        if len(indices) < n:
            raise ValueError(f"Not enough samples for label {label} (required {n}, available {len(indices)}).")

        selected_indices = np.random.choice(indices, size=n, replace=False)
        sampled_data.append(data[selected_indices])
        sampled_labels.append(labels[selected_indices])

    return np.vstack(sampled_data), np.vstack(sampled_labels)

# Function to visualize PCA of fusion vectors
def visualize_pca(features, labels):
    """
    Visualizza i vettori di fusione in 2D usando PCA.

    Args:
        features (np.array): Vettori di fusione.
        labels (np.array): Etichette corrispondenti ai vettori.
    """
    if labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    # Riduzione dimensionale con PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))

    # Creare la mappatura colori con etichette leggibili
    unique_labels = np.unique(labels)
    label_names = [label_mapping[label] for label in unique_labels]

    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # Creare una legenda personalizzata con i nomi delle classi
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_mapping[label], 
                          markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10)
               for label in unique_labels]

    plt.legend(handles=handles, title="Activities",  loc="upper right")
    plt.title("Distribution of Attention Fusion Vectors (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Function for transfer learning
def perform_transfer_learning(model, test_data, test_labels, fraction=0.1):
    """
    Fine-tunes the model using a fraction of test data.

    Args:
        model (tf.keras.Model): Pretrained MACTNet model.
        test_data (np.array): Test dataset features.
        test_labels (np.array): Test dataset labels.
        fraction (float): Fraction of the dataset to use for training.

    Returns:
        tf.keras.Model: Fine-tuned model.
    """
    fraction = retrain_rate
    X_10, X_rest, y_10, y_rest = train_test_split(
        test_data,
        test_labels,
        test_size=(1 - fraction),
        stratify=test_labels,
        random_state=42
    )
    tf = fraction *100
    ttf = (1 - fraction)*100
    print(f"Shape subset {tf}%:", X_10.shape, y_10.shape)
    print(f"Shape rest {ttf}%:", X_rest.shape, y_rest.shape)

    train_shot = create_tf_dataset(X_10, y_10, batch_size=batch_size, shuffle=True)
    val_shot = create_tf_dataset(X_rest, y_rest, batch_size=batch_size)
    performance_callback = TrainingPerformanceCallback()
    history = model.fit(
        train_shot,
        validation_data=val_shot,
        epochs=retrain_epochs,
        callbacks=[performance_callback]
    )

    return model

# Entry point
if __name__ == "__main__":

    model_test = tf.keras.models.load_model("model.keras")
    model_test.summary()

    data_set = [test_dataset]
    label_encoder = create_label_encoder(activity_labels)
    test_set_data, test_set_label = preprocess_datasets(data_set,
                                                        sequence_length,
                                                        overlap,
                                                        ant_drop,
                                                        label_encoder,
                                                        compute_module=True,
                                                        denoise=True)
    
    model_test.evaluate_model(test_set_data, test_set_label, activity_labels)
    model_test = perform_transfer_learning(model_test, test_set_data, test_set_label)
    model_test.evaluate_model(test_set_data, test_set_label, activity_labels)


