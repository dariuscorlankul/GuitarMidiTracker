import torch
from torch.utils.data import Dataset
from torch import nn
from scipy.io import wavfile
import numpy as np


class GuitarDataSet(Dataset):
    def __init__(self, wav_file_paths, label_file_paths, fs=48000, num_notes=94, midi_start=21, Q=100):
        self.wav_paths = wav_file_paths
        self.label_paths = label_file_paths
        self.fs = fs
        self.num_notes = num_notes
        self.midi_start = midi_start

        note_frequencies = 440.0 * (2.0 ** ((np.arange(midi_start, midi_start + num_notes) - 69) / 12.0))
        self.g_r, self.g_i = self._init_resonator_bank(note_frequencies, fs, Q)
        
    def _init_resonator_bank(frequencies, fs, Q=100):
        """
        Pre-calculates the static coefficients for the filter bank.
        """
        g_r = []
        g_i = []
        
        for fc in frequencies:
            theta = 2 * np.pi * fc / fs
            # r can be customized per note to get a Constant-Q or log scale layout
            r = np.exp(-np.pi * fc / (Q * fs)) 
            
            g_r.append(r * np.cos(theta))
            g_i.append(r * np.sin(theta))
            
        return np.array(g_r), np.array(g_i)

    def _process_audio_to_spectral(self, wav_path):
        sample_rate, audio_data = wavfile.read(wav_path)

        input_signal = audio_data.astype(np.float32) / np.max(np.abs(audio_data))


        #Dummy state variables
        I_state = np.zeros(self.num_notes)
        Q_state = np.zeros(self.num_notes)

        #Predefine matrix
        raw_spectral_matrix = np.zeros((len(input_signal), self.num_notes))

        for sample_id, sample in enumerate(input_signal):
            I_new = sample + (self.g_r * I_state - self.g_i * Q_state)
            Q_new = (self.g_r * Q_state + self.g_i * I_state)

            raw_spectral_matrix[sample_id] = np.sqrt(I_new**2 + Q_new**2)

            I_state = I_new
            Q_state = Q_new

            normalized_matrix = np.zeros_like(raw_spectral_matrix)

            noise_floor = 0.05

            for i, frame in enumerate(raw_spectral_matrix):
                max_val = np.max(frame)
                if max_val >= noise_floor:
                    normalized_matrix[i] = frame/max_val

        #From 0,1 to -1,1 as thats what the model needs
        normalized_matrix  = (normalized_matrix * 2.0) - 1.0

        return normalized_matrix

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, id):
        spectral_matrix = self.process_audio_to_spectral(self.wav_paths[id])

        labels_matrix = np.load(self.label_paths[id])

        num_blocks = min(len(spectral_matrix), len(labels_matrix)) - 1

        file_inputs = []
        file_targets = []

        for block_id in range(num_blocks):
            past_frame = spectral_matrix[block_id]
            current_frame = spectral_matrix[block_id + 1]

            neural_input = np.concatenate((past_frame, current_frame), axis=0)

            file_inputs.append(neural_input)
            file_targets.append(labels_matrix[block_id + 1])

            return (torch.tensor(np.array(file_inputs), dtype=torch.float32),
                    torch.tensor(np.array(file_targets), dtype=torch.float32))
 