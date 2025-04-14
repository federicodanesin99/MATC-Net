#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the MACTNet model using parameters from config.py.
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_datasets, create_tf_dataset, create_label_encoder
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
    epochs,
    training_dataset
)

if __name__ == "__main__":
    # Load dataset
    print_parameters()
    
    data_set = [training_dataset]
    print("Preprocessing datasets...")
    
    
    label_encoder = create_label_encoder(activity_labels)

    training_data, training_labels = preprocess_datasets(data_set, 
                                                         sequence_length, 
                                                         overlap, 
                                                         ant_drop, 
                                                         label_encoder, 
                                                         compute_module=True,
                                                         denoise=True)

    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        training_data, training_labels, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = MACTNet(
        num_classes=num_classes,
        time_encoder_params=time_encoder_params,
        channel_encoder_params=channel_encoder_params
    )
    x = tf.random.normal((32, sequence_length, num_features))
    out = model(x)
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    time.sleep(10)
     # Train the model with the performance callback
    print("Starting training...")
    performance_callback = TrainingPerformanceCallback()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[performance_callback],
    )

    # Salva il modello completo
    model.save("model.keras")
    print("Model weights saved.")

    # Evaluate the model
    print("Evaluating the model...")
    model.evaluate_model(X_val, y_val, activity_labels)

