#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import scipy.io as sio
from os import listdir
import pickle
from os import path
import sys
import re


# "HDA_dataset/AR-9b/"
# "HDA_dataset/AR-9c/"
# "HDA_dataset/AR-9a/"
# "HDA_dataset/AR-5b/"
# "HDA_dataset/AR-5a/"
# "HDA_dataset/AR-4a/"
# "HDA_dataset/AR-2a/"
# "HDA_dataset/AR-3a/"
# "HDA_dataset/AR-3b/"
# "HDA_dataset/AR-8b/"
# "HDA_dataset/AR-8a/"
# "HDA_dataset/AR-1e/"
# "HDA_dataset/AR-1d/"
# "HDA_dataset/AR-7a/"
# "HDA_dataset/AR-1c/"
# "HDA_dataset/AR-1b/"
# "HDA_dataset/AR-1a/"

 # data_set = ["AR1a","AR1b", "AR1c", "AR3a", "AR3b","AR4a", "AR5a", "AR8a", "AR8b", "AR9a", "AR9b"]

# sys.argv = [
#   'CSI_phase_sanitization_signal_preprocessing.py ',  # Dummy script name
#    'HDA_dataset/AR-3b/',  # Argument for 'dir'
#    '1',  # Argument for 'all_dir', set to 1 for true
#    'experiment_file_name',  # Argument for 'name'
#    '1',  # Argument for 'nss', e.g., 2 spatial streams
#    '4',  # Argument for 'ncore', e.g., 4 cores
#    '0'  # Argument for 'start_idx', start index
# ]

# Dizionario per mappare le lettere delle attività
activity_map = {
    'W': 'Walking',
    'R': 'Running',
    'J': 'Jumping',
    'L': 'Sitting still',
    'S': 'Standing still',
    'C': 'Sitting down/Standing up',
    'H': 'Doing arm exercises',
    'E': 'Empty'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('start_idx', help='Idx where start processing for each stream', type=int)
    args = parser.parse_args()

    exp_dir = args.dir
    names = []
    preprocessed_signal = []
    activity_list = []
    if args.all_dir:
        all_files = listdir(exp_dir)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        
        # Estrai il suffisso dell'attività con una regex che trova una lettera e un numero opzionale alla fine
        match = re.search(r'_([A-Z])\d*', name)
        if match:
            activity_suffix = match.group(1)
            print("match activity suffix : ",activity_suffix)
            activity = activity_map.get(activity_suffix, "Unknown activity")
        else:
            activity = "Unknown activity"
        
        name_file = './phase_processing/signal_' + name + '.txt'
        if path.exists(name_file):
            print('Already processed')
            continue

        csi_buff_file = exp_dir + name + ".mat"
        csi_buff = sio.loadmat(csi_buff_file)
        csi_buff = (csi_buff['csi_buff'])
        print("shape before fft :", csi_buff.shape)
        csi_buff = np.fft.fftshift(csi_buff, axes=1)
        print("shape after fft :", csi_buff.shape)
        delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)

        delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core

        start = args.start_idx  # 1000
        end = int(np.floor(csi_buff.shape[0]/n_tot))
        signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)
        
        # Seleziona l'antenna di riferimento (indice 0, ma può essere modificato)
        reference_antenna = 0
        
        for stream in range(0, n_tot):
            signal_stream = csi_buff[stream:end * n_tot + 1:n_tot, :][start:end, :]
            signal_stream[:, 64:] = -signal_stream[:, 64:]
        
            signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
            mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
            H_m = signal_stream / mean_signal
            
            # Sanitizzazione: normalizza il CFR usando il coniugato complesso dell'antenna di riferimento
            # if stream != reference_antenna:
            #     H_m = H_m * np.conj(signal_complete[:, :, reference_antenna].T)
            # plot_cfr_1_stream(H_m.T, stream, activity)
            signal_complete[:, :, stream] = H_m.T
            
        activity_list.append(activity)
        signal_complete_transpose = np.transpose(signal_complete, (1, 0, 2))
        
        print(f"Name file : {name_file}, Activity : {activity} - Input shape : {signal_complete_transpose.shape}")
        preprocessed_signal.append( { 'activity' : activity,
                                     'signal': signal_complete_transpose
            }
            )
        
        # name_file = './phase_processing/signal_' + name + '.txt'
        # with open(name_file, "wb") as fp:  # Pickling
        #     pickle.dump(signal_complete, fp)
            
    name_preprocessed_list = './preprocessed_signal/preprocessed_list_' +name[:4] + '.txt'
    with open(name_preprocessed_list, "wb") as fp:  # Pickling
        pickle.dump(preprocessed_signal, fp)