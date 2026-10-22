import os
import sys
import numpy as np
import json
import joblib
import traceback
import math

# Smanji TensorFlow logove
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import librosa

# Parametri moraju odgovarati treniranju!
TARGET_SR = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_CHROMA = 12
EXPECTED_SEGMENT_LENGTH_SEC = 5
EXPECTED_NUM_FRAMES = math.ceil(EXPECTED_SEGMENT_LENGTH_SEC * TARGET_SR / HOP_LENGTH)

def extract_and_process_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        if len(y) < N_FFT:
            raise ValueError("Audio file is too short")

        # Ekstrakcija značajki s istim parametrima kao u tt.py
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH)
        tempo_arr = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
        tempo = tempo_arr[0] if tempo_arr.size > 0 else 0.0
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)

        # Padding/truncation na isti način kao u tt.py
        def process_feature(feature):
            if feature.shape[1] > EXPECTED_NUM_FRAMES:
                return feature[:, :EXPECTED_NUM_FRAMES]
            elif feature.shape[1] < EXPECTED_NUM_FRAMES:
                return np.pad(feature, ((0,0), (0, EXPECTED_NUM_FRAMES - feature.shape[1])), mode='constant')
            return feature

        mfcc = process_feature(mfcc)
        chroma = process_feature(chroma)
        spectral_centroid = process_feature(spectral_centroid)
        spectral_bandwidth = process_feature(spectral_bandwidth)
        spectral_contrast = process_feature(spectral_contrast)
        zcr = process_feature(zcr)

        # Dodaj batch dimenziju kao u evaluaciji
        mfcc_batch = np.expand_dims(mfcc, axis=0)
        chroma_batch = np.expand_dims(chroma, axis=0)
        centroid_batch = np.expand_dims(spectral_centroid, axis=0)
        bandwidth_batch = np.expand_dims(spectral_bandwidth, axis=0)
        contrast_batch = np.expand_dims(spectral_contrast, axis=0)
        zcr_batch = np.expand_dims(zcr, axis=0)

        # Funkcija za statistike (identična treningu)
        def extract_stats(features):
            mean_vals = np.mean(features, axis=2)
            std_vals = np.std(features, axis=2)
            max_vals = np.max(features, axis=2)
            min_vals = np.min(features, axis=2)
            median_vals = np.median(features, axis=2)
            return np.column_stack([mean_vals, std_vals, max_vals, min_vals, median_vals])

        # Izračun statistika i redoslijed identičan batch evaluaciji!
        mfcc_stats = extract_stats(mfcc_batch)[0]
        chroma_stats = extract_stats(chroma_batch)[0]
        centroid_stats = extract_stats(centroid_batch)[0]
        bandwidth_stats = extract_stats(bandwidth_batch)[0]
        zcr_stats = extract_stats(zcr_batch)[0]
        contrast_stats = extract_stats(contrast_batch)[0]

        # Kombiniraj sve značajke u ISTOM REDOSLIJEDU kao batch evaluacija!
        features = np.concatenate([
            mfcc_stats.flatten(),
            chroma_stats.flatten(),
            centroid_stats.flatten(),
            bandwidth_stats.flatten(),
            zcr_stats.flatten(),
            contrast_stats.flatten(),
            [tempo]
        ])

        # Debug: provjeri broj značajki
        # print("Broj značajki:", len(features))

        return features.reshape(1, -1)

    except Exception as e:
        print(f"Error processing features: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            print("Usage: python predict_genre.py <audio_file_path> <model_type>")
            sys.exit(1)

        audio_path = sys.argv[1]
        model_type = sys.argv[2].lower()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))

        scaler = joblib.load(os.path.join(base_dir, 'scaler.save'))
        with open(os.path.join(base_dir, 'mappings.json'), 'r') as f:
            mappings = json.load(f)
        
        # Učitaj odgovarajući model
        if model_type == "dnn":
            model = tf.keras.models.load_model(os.path.join(base_dir, 'genre_classifier.keras'))
            scaler = joblib.load(os.path.join(base_dir, 'scaler.save'))
            is_keras = True
        elif model_type == "rf":
            model = joblib.load(os.path.join(base_dir, 'rf_model.save'))
            scaler = joblib.load(os.path.join(base_dir, 'scaler.save'))
            is_keras = False
        elif model_type == "knn":
            model = joblib.load(os.path.join(base_dir, 'knn_model.save'))
            scaler = joblib.load(os.path.join(base_dir, 'scaler.save'))
            is_keras = False
        elif model_type == "svm":
            model = joblib.load(os.path.join(base_dir, 'svm_model.save'))  # Pipeline s scalerom unutra
            is_keras = False
        else:
            print("Nepoznat model_type! Koristi 'dnn', 'rf', 'knn' ili 'svm'.")
            sys.exit(1)

        # Ekstrakcija značajki
        features = extract_and_process_features(audio_path)
        if features is None:
            sys.exit(1)

        # Skaliranje i predikcija
        if model_type == "svm":
            # NE SKALIRAJ za SVM! Pipeline to radi interno
            predicted_index = int(model.predict(features)[0])
        else:
            # Skaliraj za sve ostale modele
            features_scaled = scaler.transform(features)
            if is_keras:
                prediction = model.predict(features_scaled, verbose=0)
                predicted_index = int(np.argmax(prediction))
            else:
                predicted_index = int(model.predict(features_scaled)[0])

        # Mapiranje indeksa u žanr
        if isinstance(mappings, dict):
            predicted_genre = mappings.get(str(predicted_index), mappings[predicted_index])
        else:
            predicted_genre = mappings[predicted_index]

        print(str(predicted_genre), flush=True)

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


