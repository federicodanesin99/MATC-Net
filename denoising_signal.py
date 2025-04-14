#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:16:38 2025

@author: federicodanesin

"""
import pywt
import numpy as np
import pickle
import gc

def hampel_filter_3d(input_matrix, window_size, n_sigmas=3):
    """
    Applica il filtro Hampel a un segnale tridimensionale (time x subcarrier x antenna).
    
    Args:
        input_matrix (np.ndarray): Segnale di input di dimensioni (time, subcarrier, antenna).
        window_size (int): Dimensione della finestra temporale.
        n_sigmas (float): Soglia in unità di deviazione standard.

    Returns:
        np.ndarray: Segnale filtrato.
    """
    time, subcarriers, antennas = input_matrix.shape
    filtered_matrix = np.zeros_like(input_matrix)  
    k = 1.4826  

    for sc in range(subcarriers): 
        print("ciii")
        for ant in range(antennas): 
            signal = input_matrix[:, sc, ant]
            for t in range(time):

                start_time = max(0, t - window_size)
                end_time = min(time, t + window_size + 1)
                window = signal[start_time:end_time]
                x0 = np.nanmedian(window)
                mad = k * np.nanmedian(np.abs(window - x0))

                if np.abs(signal[t] - x0) > n_sigmas * mad:
                    filtered_matrix[t, sc, ant] = x0 
                else:
                    filtered_matrix[t, sc, ant] = signal[t]  

    return filtered_matrix

def wavelet_denoising_subcarrier(signal_module, wavelet='sym4', level=None, attenuate_levels=6, threshold_scale=1.0):
    """
    Applica il denoising wavelet lungo la dimensione temporale per ogni subcarrier e antenna.

    Args:
        signal_module (np.ndarray): Segnale con dimensioni (tempo, subcarriers, antenne). Supporta segnali complessi.
        wavelet (str): Tipo di wavelet da usare.
        level (int): Numero di livelli di decomposizione wavelet. Se None, viene calcolato automaticamente.
        attenuate_levels (int): Numero di livelli di dettaglio da attenuare.
        threshold_scale (float): Fattore per scalare la soglia.

    Returns:
        np.ndarray: Segnale denoised con le stesse dimensioni di input.
    """
    num_time, num_subcarriers, num_antennas = signal_module.shape

    # Calcolo automatico del livello di decomposizione se non specificato
    if level is None:
        max_level = pywt.dwt_max_level(num_time, pywt.Wavelet(wavelet).dec_len)
        level = min(max_level, attenuate_levels)
    
    # Verifica compatibilità attenuate_levels e livello massimo
    if attenuate_levels > level:
        raise ValueError(f"`attenuate_levels` ({attenuate_levels}) non può essere maggiore di `level` ({level}).")
        
    print(f"Livello calcolato : {level}")

    denoised_signal_module = np.zeros_like(signal_module, dtype=signal_module.dtype)

    # Itera su subcarrier e antenne
    for antenna in range(num_antennas):
        for subcarrier in range(num_subcarriers):
            # Estrai la serie temporale
            time_series = signal_module[:, subcarrier, antenna]

            # Decomposizione wavelet
            coeffs = pywt.wavedec(time_series, wavelet, mode='symmetric', level=level)

            # Calcolo della soglia basata sui coefficienti di dettaglio
            detail_coeffs = np.concatenate(coeffs[1:attenuate_levels + 1])
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745
            threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(time_series)))

            # Applica la soglia ai coefficienti di dettaglio
            for i in range(1, attenuate_levels + 1):
                coeffs[i] = pywt.threshold(coeffs[i], value=threshold, mode='soft')

            # Ricostruzione del segnale
            denoised_time_series = pywt.waverec(coeffs, wavelet, mode='symmetric')

            # Allineamento della lunghezza del segnale ricostruito
            if len(denoised_time_series) > num_time:
                denoised_time_series = denoised_time_series[:num_time]
            elif len(denoised_time_series) < num_time:
                denoised_time_series = np.pad(denoised_time_series, (0, num_time - len(denoised_time_series)), mode='edge')

            # Assegna il segnale denoised al risultato
            denoised_signal_module[:, subcarrier, antenna] = denoised_time_series

    return denoised_signal_module

def denoising_function(preprocessed_signal):
    denoised_list = []    
    for i, signal_dict in enumerate(preprocessed_signal):
        signal = signal_dict.get('signal')
        activity = signal_dict.get('activity')
        print(f"Segnale: {signal.shape}, Attività: {activity}")
        print(f"Wavelet denoising...")
        signal_denoised = wavelet_denoising_subcarrier(signal)
        print(f"Module extraction...")
        signal_denoised = np.abs(signal_denoised)
        print(f"Hampel Filter...")
        signal_denoised = hampel_filter_3d(signal_denoised, window_size=5, n_sigmas=3)
        denoised_dict = { 'signal': signal_denoised,
                          'activity': activity 
                        }
        denoised_list.append(denoised_dict)
        
    return denoised_list

def processing_pipeline(data_set):
    """Elabora i dataset forniti, calcola il modulo dei segnali e ritorna una lista di moduli con attività e nomi dataset,
    insieme a un DataFrame aggregato dei risultati.

    Args:
        data_set (list): Lista di nomi dei dataset da elaborare.
        save_intermediate (bool): Se True, salva i dati intermedi su disco.
        intermediate_file (str): Nome del file per salvare i dati intermedi.

    Returns:
        list: Lista di dizionari con chiavi 'module', 'activity', e 'name_data' per ogni dataset.
        DataFrame: DataFrame con i campi 'features' e 'label' aggregati per ogni dataset.
    """
    for data_name in data_set:
        # Carica i dati preelaborati
        name_preprocessed_list = f'./preprocessed_signal/preprocessed_list_{data_name}.txt'
        print(f"Processo il file: {name_preprocessed_list}")
        with open(name_preprocessed_list, "rb") as fp:
            preprocessed_signal = pickle.load(fp)
        denoised_signal = denoising_function(preprocessed_signal)
        
        name_denoised_list = './denoised_signal/denoised_list_' + data_name + '.txt'
        with open(name_denoised_list, "wb") as fp:  # Pickling
            pickle.dump(denoised_signal, fp)

        del preprocessed_signal
        gc.collect()


if __name__ == '__main__':
    
    # data_set = ["AR1b", "AR1c", "AR3a", "AR3b","AR4a", "AR5a", "AR8a", "AR8b", "AR9a", "AR9b"]
    data_set = ["AR4a"]
    processing_pipeline(data_set)
    






