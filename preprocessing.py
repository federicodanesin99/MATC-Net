#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:38:43 2025

@author: federicodanesin
"""

# Main imports
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import random
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from denoising_signal import wavelet_denoising_subcarrier
from collections import Counter
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
    label_mapping
)


def drop_channels(signal, num_channels_to_drop):
    """
    Randomly drops a specified number of channels from the signal.

    Args:
        signal (np.ndarray): Input signal of shape (time, subcarriers, channels).
        num_channels_to_drop (int): Number of channels to randomly set to zero.

    Returns:
        np.ndarray: Signal with selected channels dropped.

    Raises:
        ValueError: If the number of channels to drop exceeds the available channels.
    """
    total_channels = signal.shape[-1]
    if num_channels_to_drop > total_channels:
        raise ValueError("Cannot drop more channels than available.")

    channels_to_drop = random.sample(range(total_channels), num_channels_to_drop)
    channels_to_keep = [i for i in range(total_channels) if i not in channels_to_drop]
    return signal[:, :, channels_to_keep]

def segment_signal(signal_data, seq_len=100, overlap=50, drop_channels_num=0, compute_module=False, denoise=False):
    """
    Segments the signal into overlapping windows and optionally computes the module.

    Args:
        signal_data (list): List of dictionaries with keys 'signal' and 'activity'.
        seq_len (int): Length of each segment.
        overlap (int): Number of overlapping samples between consecutive segments.
        drop_channels_num (int): Number of channels to drop.
        compute_module (bool): Whether to compute the module of the signal.

    Returns:
        list: Segmented signal data with associated activities.
    """
    segmented_data = []

    # Ensure the overlap does not exceed the segment length
    if overlap >= seq_len:
        raise ValueError("The overlap must be smaller than the sequence length (seq_len).")

    step_size = seq_len - overlap
    for entry in signal_data:
        signal = entry['signal']
        activity = entry['activity']
        print(f"Signal shape: {signal.shape}, Activity: {activity}")
        if denoise:
            print("Denoising...")
            signal = wavelet_denoising_subcarrier(signal)
        if drop_channels_num > 0:
            signal = drop_channels(signal, drop_channels_num)
        print("Segmentation...\n")
        for start in range(0, signal.shape[0] - seq_len + 1, step_size):
            segment = signal[start:start + seq_len]
            if compute_module:
                segment = np.abs(segment)
            segmented_data.append({'segment': segment, 'activity': activity})

    return segmented_data


def process_data_pipeline(data_names, seq_len, overlap=50, drop_channels_num=0, compute_module=False, denoise=False):
    """
    Processes multiple datasets by applying segmentation and optional transformations.

    Args:
        data_names (list): List of dataset names to process.
        seq_len (int): Length of each segment.
        overlap (int): Number of overlapping samples between consecutive segments.
        drop_channels_num (int): Number of channels to drop during preprocessing.
        compute_module (bool): Whether to compute the module of the signal.

    Returns:
        list: Processed data across all datasets.
    """
    all_processed_data = []

    for name in data_names:
        print(f"Processing dataset: {name}")
        if compute_module: 
            with open(f'./preprocessed_signal/preprocessed_list_{name}.txt', 'rb') as file:
                signal_data = pickle.load(file)
        else:
            with open(f'./denoised_signal/denoised_list_{name}.txt', 'rb') as file:
                signal_data = pickle.load(file)

        processed_segments = segment_signal(
            signal_data,
            seq_len=seq_len,
            overlap=overlap,
            drop_channels_num=drop_channels_num,
            compute_module=compute_module,
            denoise=denoise
        )
        all_processed_data.extend(processed_segments)
        del signal_data
        gc.collect()

    return all_processed_data

def normalize_data(X):
    """
    Normalizes the dataset using StandardScaler if specified.

    Args:
        X (np.ndarray): Input dataset to normalize.

    Returns:
        np.ndarray: Normalized dataset.
    """
    # mean = np.mean(X)
    # std = np.std(X)
    # X_normalized = (X - mean) / std
    # print("Forma del dataset normalizzato X:", X_normalized.shape)
    print("Normalizing...")
    scaler = StandardScaler()
    X_flattened = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flattened)
    X_scaled = X_scaled.reshape(X.shape)
    return X_scaled


def prepare_dataset(processed_data, label_encoder, normalize=True, num_classes=8):
    """
    Prepares the dataset for training, including normalization and label encoding.

    Args:
        processed_data (list): Segmented and processed data.
        label_encoder (LabelEncoder): Pre-trained LabelEncoder for activity labels.
        normalize (bool): Whether to normalize the input data.
        num_classes (int): Total number of activity classes.

    Returns:
        tuple: (X_normalized, y_one_hot), where X is the normalized input data and y is one-hot encoded labels.
    """
    X = np.array([entry['segment'] for entry in processed_data])
    y = np.array([entry['activity'] for entry in processed_data])
    print(f"Lunghezza di X {X.shape} e y {y.shape}")
    y_encoded = label_encoder.transform(y)
    y_one_hot = to_categorical(y_encoded, num_classes=num_classes)
    
    X_normalized = normalize_data(X)

    return X_normalized, y_one_hot

def create_label_encoder(class_labels):
    """
    Creates and fits a LabelEncoder for the given class labels.

    Args:
        class_labels (list): List of class labels.

    Returns:
        LabelEncoder: Fitted label encoder.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)
    return label_encoder

def reshape_data(X):
    """
    Reshapes the dataset by concatenating subcarrier and antenna dimensions.

    Args:
        X (np.ndarray): Input dataset of shape (samples, time_steps, subcarriers, antennas).

    Returns:
        np.ndarray: Reshaped dataset of shape (samples, time_steps, subcarriers * antennas).
    """
    time_steps, subcarriers, antennas = X.shape[1], X.shape[2], X.shape[3]
    return X.reshape(X.shape[0], time_steps, subcarriers * antennas)

def create_tf_dataset(X, y, batch_size, shuffle=True):
    """
    Creates a TensorFlow dataset for efficient training and evaluation.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def print_class_distribution(y, label_mapping):
    """
    Prints the distribution of classes in the dataset.

    Args:
        y (np.ndarray): One-hot encoded labels.
        label_mapping (dict): Mapping of class indices to human-readable labels.
    """
    class_counts = Counter(np.argmax(y, axis=1))
    for label, count in class_counts.items():
        print(f"{label_mapping[label]}: {count}")

def preprocess_datasets(data_names, seq_len, overlap, drop_channels_num, label_encoder, compute_module=False, denoise=False, reshape=True):
    """
    Main preprocessing function to handle segmentation, normalization, and reshaping.

    Args:
        data_names (list): List of dataset names to preprocess.
        seq_len (int): Length of each segment.
        overlap (int): Number of overlapping samples.
        drop_channels_num (int): Number of channels to drop during preprocessing.
        label_encoder (LabelEncoder): Pre-trained LabelEncoder for activity labels.

    Returns:
        tuple: (X_reshaped, y), where X is the reshaped input data and y is one-hot encoded labels.
    """
    processed_data = process_data_pipeline(data_names, seq_len, overlap, drop_channels_num, compute_module, denoise)
    X, y = prepare_dataset(processed_data, label_encoder, normalize=False)
    if reshape:
        X = reshape_data(X)
    return X, y

def plot_sample_sequences(training_data, training_labels, subcarrier_idx, antenna_idx, label_encoder):
    """
    Crea un grafico per un campione casuale per ogni etichetta nel dataset.
    :param training_data: Array con forma (N, 100, 242, 4) -> N campioni, 100 timestep, 242 subcarriers, 4 antenne.
    :param training_labels: Array con forma (N, 8) -> Etichette one-hot.
    :param subcarrier_idx: Indice della subcarrier da visualizzare.
    :param antenna_idx: Indice dell'antenna da visualizzare.
    """
    # Converti le etichette one-hot in valori scalari
    label_indices = np.argmax(training_labels, axis=1)
    # Prepara i grafici
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(label_indices)

    for i, label in enumerate(unique_labels):
        samples_with_label = np.where(label_indices == label)[0]
        sample_idx = random.choice(samples_with_label)
        sample = training_data[sample_idx]  
        sequence = sample[:, subcarrier_idx, antenna_idx]
        plt.subplot(2, 4, i + 1)  # Layout 2 righe x 4 colonne
        plt.plot(sequence, label=f"Label: {label_mapping[label]}")
        plt.title(label_encoder.inverse_transform([label])[0])
        plt.xlabel("Timestep")
        plt.ylabel("Amplitude")
        plt.ylim(-5,5)
        plt.grid(True)
        plt.tight_layout()           
    plt.suptitle(f"Sequences for subcarrier {subcarrier_idx} and Antenna {antenna_idx}", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == '__main__':
    data_set = ["AR1a", "AR1b", "AR1c", "AR3a", "AR3b","AR4a", "AR5a", "AR9a", "AR9b"]
    print("Preprocessing datasets...")
    label_encoder = create_label_encoder(activity_labels)
    training_data, training_labels = preprocess_datasets(
        data_set, 300, overlap, ant_drop, label_encoder, compute_module=True, reshape=False
    )
    print_class_distribution(training_labels, label_mapping)
    subcarrier_idx = 200
    antenna_idx = 0
    plot_sample_sequences(training_data, training_labels, subcarrier_idx, antenna_idx, label_encoder)
    
