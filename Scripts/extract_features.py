import librosa
import numpy as np
import json
import math
import sys
import os  
import traceback



INPUT_BASE_DIR = 'podaci/train'  

OUTPUT_JSON_PATH = 'podaci/train.json'  

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.au')


TARGET_SR = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_CHROMA = 12
EXPECTED_SEGMENT_LENGTH_SEC = 5


EXPECTED_NUM_FRAMES = math.ceil(EXPECTED_SEGMENT_LENGTH_SEC * TARGET_SR / HOP_LENGTH)

print(f"Using Sample Rate: {TARGET_SR} Hz")
print(f"FFT Window: {N_FFT}, Hop Length: {HOP_LENGTH}")
print(f"Expected feature frames per clip: {EXPECTED_NUM_FRAMES}")



def extract_features(input_dir, output_json_path):
    """
    Extracts MFCC, Chroma, Tempo, Spectral Centroid, Spectral Bandwidth,
    Spectral Contrast, and Zero-Crossing Rate features from audio files,
    pads/truncates sequences to a fixed length, and saves them to a JSON file.

    Args:
        input_dir (str): Path to the directory containing subdirectories
                         for each genre/class (e.g., 'classical', 'disco', 'noise').
        output_json_path (str): Path to save the resulting JSON file.
    """
    data = {
        "mappings": [],            # List of genre/class names
        "labels": [],              # Corresponding numerical labels (indices in mappings)
        "mfcc": [],                # Timbre (sequence)
        "chroma": [],              # Harmony (sequence)
        "tempo": [],               # Overall rhythm speed (single value per clip)
        "spectral_centroid": [],   # Brightness (sequence)
        "spectral_bandwidth": [],  # Spectral width (sequence)
        "spectral_contrast": [],   # Peak-valley difference (sequence)
        "zcr": [],                 # Percussiveness/Noisiness (sequence)
        "filenames": []            # Relative path of the processed file
    }
    skipped_files = []
    processed_files_count = 0

    print(f"\nStarting feature extraction from: {input_dir}")
    print(f"Target output file: {output_json_path}")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' not found.")
        return


    try:

        genre_dirs = sorted([d for d in os.listdir(input_dir)
                             if os.path.isdir(os.path.join(input_dir, d))])
        if not genre_dirs:
            print(f"[ERROR] No subdirectories found in '{input_dir}'. Expecting genre folders.")
            return
        data["mappings"] = genre_dirs
        print(f"Found genre/class folders: {data['mappings']}")
    except OSError as e:
        print(f"[ERROR] Could not list directories in '{input_dir}': {e}")
        return

    for genre_name in data["mappings"]:
        label = data["mappings"].index(genre_name)
        dirpath = os.path.join(input_dir, genre_name)

        print(f"\nProcessing Folder: {genre_name} (Label: {label})")

        try:
            filenames = [f for f in os.listdir(dirpath)
                         if os.path.isfile(os.path.join(dirpath, f))]
        except OSError as e:
            print(f"  [ERROR] Could not list files in '{dirpath}': {e}. Skipping folder.")
            continue  


        for f in filenames:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                file_path = os.path.join(dirpath, f)

                relative_file_path = os.path.join(genre_name, f)

                try:

                    y, sr = librosa.load(file_path, sr=TARGET_SR)

                    if len(y) < N_FFT:
                        print(f"  [WARNING] Audio file {relative_file_path} is too short ({len(y)} samples). Skipping.")
                        skipped_files.append(relative_file_path + " (Reason: Too short)")
                        continue


                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                                n_fft=N_FFT, hop_length=HOP_LENGTH)
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA,
                                                         n_fft=N_FFT, hop_length=HOP_LENGTH)


                    tempo_array = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)

                    tempo = tempo_array[0] if tempo_array.size > 0 else 0.0

                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
                    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
                    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)

                    current_frames = mfcc.shape[1]

              
                    if current_frames == 0:
                        print(f"  [WARNING] Feature extraction resulted in 0 frames for {relative_file_path}. Skipping.")
                        skipped_files.append(relative_file_path + " (Reason: 0 frames extracted)")
                        continue

                    if current_frames > EXPECTED_NUM_FRAMES:
                        mfcc = mfcc[:, :EXPECTED_NUM_FRAMES]
                        chroma = chroma[:, :EXPECTED_NUM_FRAMES]
                        spectral_centroid = spectral_centroid[:, :EXPECTED_NUM_FRAMES]
                        spectral_bandwidth = spectral_bandwidth[:, :EXPECTED_NUM_FRAMES]
                        spectral_contrast = spectral_contrast[:, :EXPECTED_NUM_FRAMES]
                        zcr = zcr[:, :EXPECTED_NUM_FRAMES]
                    elif current_frames < EXPECTED_NUM_FRAMES:
                        pad_width_frames = EXPECTED_NUM_FRAMES - current_frames
                    
                        pad_tuple = ((0, 0), (0, pad_width_frames))

                        mfcc = np.pad(mfcc, pad_width=pad_tuple, mode='constant', constant_values=0)
                        chroma = np.pad(chroma, pad_width=pad_tuple, mode='constant', constant_values=0)
                        spectral_centroid = np.pad(spectral_centroid, pad_width=pad_tuple, mode='constant', constant_values=0)
                        spectral_bandwidth = np.pad(spectral_bandwidth, pad_width=pad_tuple, mode='constant', constant_values=0)
                        spectral_contrast = np.pad(spectral_contrast, pad_width=pad_tuple, mode='constant', constant_values=0)
                        zcr = np.pad(zcr, pad_width=pad_tuple, mode='constant', constant_values=0)


                    assert mfcc.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"MFCC shape error in {f}: {mfcc.shape} vs expected ({N_MFCC},{EXPECTED_NUM_FRAMES})"
                    assert chroma.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"Chroma shape error in {f}: {chroma.shape} vs expected ({N_CHROMA},{EXPECTED_NUM_FRAMES})"
                    assert spectral_centroid.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"Centroid shape error in {f}: {spectral_centroid.shape} vs expected (1,{EXPECTED_NUM_FRAMES})"
                    assert spectral_bandwidth.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"Bandwidth shape error in {f}: {spectral_bandwidth.shape} vs expected (1,{EXPECTED_NUM_FRAMES})"
     
                    assert spectral_contrast.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"Contrast shape error in {f}: {spectral_contrast.shape} vs expected ({spectral_contrast.shape[0]},{EXPECTED_NUM_FRAMES})"
                    assert zcr.shape[1] == EXPECTED_NUM_FRAMES, \
                        f"ZCR shape error in {f}: {zcr.shape} vs expected (1,{EXPECTED_NUM_FRAMES})"


                    data["mfcc"].append(mfcc.tolist())
                    data["chroma"].append(chroma.tolist())
                    data["tempo"].append(float(tempo)) 
                    data["spectral_centroid"].append(spectral_centroid.tolist())
                    data["spectral_bandwidth"].append(spectral_bandwidth.tolist())
                    data["spectral_contrast"].append(spectral_contrast.tolist())
                    data["zcr"].append(zcr.tolist())

                    data["labels"].append(label)
                    data["filenames"].append(relative_file_path) 
                    processed_files_count += 1

 
                    if processed_files_count % 100 == 0:
                        print(f"  Processed {processed_files_count} files...")

                except Exception as e:

                    print(f"  [ERROR] Failed to process {relative_file_path}: {e}")

                    traceback.print_exc()
                    skipped_files.append(relative_file_path + f" (Error: {e})")


    try:
        print(f"\nSaving extracted features to {output_json_path}...")

        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True) 

        with open(output_json_path, "w") as fp:
            json.dump(data, fp, indent=4)
        print("Save complete.")
    except Exception as e:
        print(f"[ERROR] Could not write JSON file '{output_json_path}': {e}")
        traceback.print_exc()

    print(f"\n--- Feature Extraction Summary ---")
    print(f"Total files processed successfully: {processed_files_count}")
    print(f"Total files skipped (errors/warnings): {len(skipped_files)}")
    if skipped_files:
        print("Skipped files list (showing up to 50):")
        limit = min(len(skipped_files), 50)
        for skipped in skipped_files[:limit]:
            print(f"  - {skipped}")
        if len(skipped_files) > limit:
            print(f"  ... and {len(skipped_files) - limit} more.")



if __name__ == "__main__":

    if not os.path.isdir(INPUT_BASE_DIR):
        print(f"[FATAL ERROR] Input directory '{INPUT_BASE_DIR}' does not exist.")
        print("Please check the INPUT_BASE_DIR variable.")
        sys.exit(1)  
    extract_features(INPUT_BASE_DIR, OUTPUT_JSON_PATH)

    print("\nScript finished.")