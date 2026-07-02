import torch
from torch import nn
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import normalize


samplerate, data = wavfile.read('data.wav')

data = data.astype(np.float32) / np.max(np.abs(data))

print(f"samplerate {samplerate}")


def init_resonator_bank(frequencies, fs, Q=50):
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

def midi_to_note_name(midi_num):
    """
    Converts a MIDI number to a standard Western musical note name (e.g., E2, A3).
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    note = notes[midi_num % 12]
    return f"{note}{octave}"

num_notes = 94

note_frequencies = 440.0 * (2.0 ** ((np.arange(21, 21 + num_notes) - 69) / 12.0))

g_r, g_i = init_resonator_bank(note_frequencies, samplerate, Q=100)

I_state = np.zeros(num_notes)
Q_state = np.zeros(num_notes)

for sample_idx, sample in enumerate(data):
    I_new =  sample + (g_r * I_state - g_i * Q_state)
    Q_new = (g_r * Q_state + g_i * I_state)


    vector = np.sqrt(I_new ** 2 + Q_new ** 2)

    vector = normalize(vector[:,np.newaxis], axis=0).ravel()

    I_state = I_new
    Q_state = Q_new

    if sample_idx % 32 == 0:
        time_ms = (sample_idx / samplerate) * 1000
        print(f"\n=======================================================")
        print(f" SNAPSHOT AT SAMPLE {sample_idx} ({time_ms:.2f} ms)")
        print(f"=======================================================")
        print(f"{'Note':<8} | {'Frequency (Hz)':<15} | {'Amplitude (Energy)':<20}")
        print("-" * 50)
        
        # Loop over the 94 channels to display the results sequentially
        for idx, (freq, amp) in enumerate(zip(note_frequencies, vector)):
            midi_num = 21 + idx
            note_name = midi_to_note_name(midi_num)
            
            # --- CHIP SELECT SHORTCUT FOR TERMINAL CLEANLINESS ---
            # Guitar note signals are concentrated. To keep the terminal from 
            # turning into an endless wall of zero text, we only print bins 
            # that pass a minimal background noise floor threshold (e.g., 0.05)
            if amp > 0.05:
                print(f"{note_name:<8} | {freq:<15.2f} | {amp:<20.4f}")
                
        # --- TESTING SAFETY BRAKE ---
        # A 1-second file at 96kHz has 3,000 blocks of 32 samples.
        # Let's break the script early after 5 print cycles so your terminal 
        # doesn't completely lock up. Remove this check when generating datasets.
        if sample_idx >= 32 * 50:
            print("\n[Brake Triggered: Exited early to save terminal view]")
            break


