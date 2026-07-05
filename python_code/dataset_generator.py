import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import midi2audio
import numpy as np
import pretty_midi
from midi2audio import FluidSynth
import scipy.io.wavfile as wav
from tqdm import tqdm


GUITAR_PROGRAMS = range(24, 31)
SOUNDFONT_PATH = 'midi_files/guitar.sf2'
_SYNTH = None


def _init_wav_worker(soundfont_path, sample_rate):
    global _SYNTH
    _SYNTH = FluidSynth(soundfont_path, sample_rate=sample_rate)


def _iter_midi_files(root_folder, limit=None):
    discovered = 0

    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames.sort()
        for filename in sorted(filenames):
            if limit is not None and discovered >= limit:
                return

            if filename.lower().endswith(('.mid', '.midi')):
                discovered += 1
                yield os.path.join(dirpath, filename)


def _extract_guitar_worker(midi_path):
    try:
        return midi_path if extract_guitar_from_file(midi_path) else None
    except (OSError, Exception) as er:
        print(f"this one was corrupted {midi_path}, detail{er}")
        return None


def _midi_to_wav_worker(midi_path):
    new_path = midi_path.replace('midi_files', 'wav_training')
    new_path = os.path.splitext(new_path)[0] + '.wav'

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    _SYNTH.midi_to_audio(midi_path, new_path)
    return new_path

def extract_guitar_from_file(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)

    guitar_tracks = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        if instrument.program in GUITAR_PROGRAMS:

            print(f'fount guitarguitar at {midi_path}')
            instrument.program = 0

            instrument.channel = 0

            guitar_tracks.append(instrument)

    if guitar_tracks:
        pm.instruments = guitar_tracks
        pm.write(midi_path)

        return True

    return False


def process_midi_directory(root_folder, workers=None, limit=None):
    directory_list = []
    midi_files = list(_iter_midi_files(root_folder, limit=limit))

    if not midi_files:
        with open('output.txt', 'w') as file:
            pass
        return directory_list

    worker_count = workers or os.cpu_count() or 1

    try:
        with mp.Pool(processes=worker_count) as pool:
            for result in tqdm(pool.imap_unordered(_extract_guitar_worker, midi_files), total=len(midi_files), desc='processing midi'):
                if result:
                    directory_list.append(result)
    except KeyboardInterrupt:
        print(directory_list)

    directory_list = sorted(directory_list)

    with open('output.txt', 'w') as file:
        for midi_path in directory_list:
            print(midi_path, file=file)

    return directory_list


def midi_to_wav(workers=None, limit=None):
    with open('output.txt') as f:
        lines = [line.strip() for line in f if line.strip()]

    if limit is not None:
        lines = lines[:limit]

    worker_count = workers or os.cpu_count() or 1

    with mp.Pool(processes=worker_count, initializer=_init_wav_worker, initargs=(SOUNDFONT_PATH, 48000)) as pool:
        for _ in tqdm(pool.imap_unordered(_midi_to_wav_worker, lines), total=len(lines), desc='audio rendering'):
            pass



def generate_single_label_matrix(midi_path, fs=48000, num_notes=94, midi_start=21):
    """
    Worker function executed in parallel.
    Converts a single MIDI file into a frame-by-frame binary matrix matching 
    the exact length of its corresponding synthesized .wav audio track.
    """
    try:
        # 1. Determine the path of the matching generated wav file
        wav_path = midi_path.replace("midi_files", "wav_training")
        wav_path = os.path.splitext(wav_path)[0] + ".wav"
        
        if not os.path.exists(wav_path):
            return (midi_path, f"Skipped: Matching WAV file does not exist at {wav_path}")
            
        # 2. Open the audio file strictly to extract its exact duration down to the individual sample
        samplerate, audio_data = wav.read(wav_path)
        total_audio_samples = len(audio_data)
        
        # 3. Pre-allocate an empty multi-label binary target matrix
        # Shape: (Total Audio Samples, 94 Note Bins)
        piano_roll_targets = np.zeros((total_audio_samples, num_notes), dtype=np.float32)
        
        # 4. Parse the MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        max_note_limit = midi_start + num_notes
        
        # Scan notes on all remaining clean guitar tracks
        for instrument in pm.instruments:
            for note in instrument.notes:
                # Confirm the pitch lands within your physical 94-note guitar frame
                if midi_start <= note.pitch < max_note_limit:
                    note_idx = note.pitch - midi_start
                    
                    # Convert absolute second timestamps to exact 48 kHz sample pointers
                    start_sample = int(note.start * fs)
                    end_sample = int(note.end * fs)
                    
                    # Hard boundaries to protect against array index overflows
                    start_sample = max(0, min(start_sample, total_audio_samples - 1))
                    end_sample = max(0, min(end_sample, total_audio_samples - 1))
                    
                    # Flag note active frames as 1.0 for BCEWithLogitsLoss compliance
                    piano_roll_targets[start_sample:end_sample, note_idx] = 1.0
                    
        # 5. Construct output destination inside a parallel "labels_training" folder tree
        label_output_path = midi_path.replace("midi_files", "labels_training")
        label_output_path = os.path.splitext(label_output_path)[0] + ".npy"
        
        # Guarantee sub-directories are generated safely across threads
        os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
        
        # Save as a highly optimized compressed NumPy file matrix
        np.save(label_output_path, piano_roll_targets)
        return (midi_path, True)
        
    except Exception as e:
        return (midi_path, f"Execution Failure: {e}")


def batch_extract_labels_multiprocess():
    if not os.path.exists("output.txt"):
        print("ERROR: 'output.txt' indexing file missing. Run directory sorting first!")
        return

    with open("output.txt") as f:
        midi_paths = [line.strip() for line in f if line.strip()][:100]

    print(f"Spawning parallel workers to extract training targets for {len(midi_paths)} files...")
    
    # Map label target files text paths back for our PyTorch Dataset constructor to ingest later
    label_indices = []
    
    with ProcessPoolExecutor() as executor:
        # Submit execution maps to our processing core slots
        futures = [executor.submit(generate_single_label_matrix, path) for path in midi_paths]
        
        for future in tqdm(as_completed(futures), total=len(midi_paths), desc="Extracting Targets"):
            midi_path, result = future.result()
            
            if result is True:
                expected_label_path = midi_path.replace("midi_files", "labels_training")
                expected_label_path = os.path.splitext(expected_label_path)[0] + ".npy"
                label_indices.append(expected_label_path)
            else:
                print(f"\n[!] Worker Error on {os.path.basename(midi_path)} | Details: {result}")
                
    # Save a clean label register log text file for tracking verification
    with open("labels_output.txt", "w") as file:
        for path in label_indices:
            print(path, file=file)
            
    print(f"\n[+] Label matrix creation finished! Matrices saved to 'training_set/labels_training/'")
            


if __name__ == "__main__":

    directory = 'training_set/midi_files/'
    #midi_to_wav(limit=50) 
    #process_midi_directory(directory)
    batch_extract_labels_multiprocess()

