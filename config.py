#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:12:50 2025

@author: federicodanesin
"""


batch_size = 32
num_antennas = 4  # supponiamo 4 antenne
subcarriers = 242
num_classes = 8   # supponiamo 8 classi
sequence_length = 100
overlap = 80
denoising = False
ant_drop = 0

num_streams = 4
seq_len = sequence_length
num_features = (num_streams - ant_drop) * subcarriers

num_layers_time = 4
num_heads_time = 8
key_dim_time = 128
embedding_dim_time = 128

num_layers_channel = 4
num_heads_channel = 8
key_dim_channel = 128
embedding_dim_channel = 128

epochs = 50
retrain_epochs = 10
retrain_rate= .3


# Possible choice for training e test dataset
# data_set = ["AR1a", "AR1b", "AR1c", "AR3a", "AR3b","AR4a", "AR5a", "AR8a", "AR8b", "AR9a", "AR9b"]

training_dataset = 'AR4a'
test_dataset = 'AR3a'

time_encoder_params = { 
    "num_layers": num_layers_time, 
    "num_filters": embedding_dim_time,
    "key_dim": key_dim_time,
    "num_heads": num_heads_time, 
    "seq_len": seq_len, 
    "embedding_dim": embedding_dim_time,
    "dropout_rate": 0.2
}

channel_encoder_params = { 
    "num_layers": num_layers_channel, 
    "num_filters": embedding_dim_channel,
    "key_dim": key_dim_channel,
    "num_heads": num_heads_channel, 
    "seq_len": seq_len, 
    "embedding_dim": embedding_dim_channel,
    "dropout_rate": 0.2
}

activity_labels = [
    "Walking", "Running", "Sitting still", "Standing still",
    "Sitting down/Standing up", "Empty", "Jumping", "Doing arm exercises"
]

label_mapping = {0: 'Walking', 1: 'Running', 2: 'Sitting still', 3: 'Standing still', 
                 4: 'Sitting down/Standing up', 5: 'Empty', 6: 'Jumping', 7: 'Doing arm exercises'}

def print_parameters():
    print("Configuration Parameters:")
    print(f"Batch size: {batch_size}")
    print(f"Number of antennas: {num_antennas}")
    print(f"Number of subcarriers: {subcarriers}")
    print(f"Number of classes: {num_classes}")
    print(f"Sequence length: {sequence_length}")
    print(f"Overlap: {overlap}")
    print(f"Denoising: {denoising}")
    print(f"Antennas dropped: {ant_drop}")
    print(f"Training epochs: {epochs}")

    print("\nTime Encoder Parameters:")
    for key, value in time_encoder_params.items():
        print(f"  {key}: {value}")

    print("\nChannel Encoder Parameters:")
    for key, value in channel_encoder_params.items():
        print(f"  {key}: {value}")

    print("\nActivity Labels:")
    for i, label in enumerate(activity_labels):
        print(f"  {i}: {label}")

# Example usage
if __name__ == "__main__":
    print_parameters()
