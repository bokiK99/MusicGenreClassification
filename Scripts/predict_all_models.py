import librosa
import numpy as np
import json
import joblib
import tensorflow as tf
import sys
import os
import traceback
import math

# Potpuno ukloni TensorFlow ispise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Parametri
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

        # Ekstrakcija značajki
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
        tempo = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)[0]

        # Padding/truncation
        def process_feature(feature):
            if feature.shape[1] > EXPECTED_NUM_FRAMES:
                return feature[:, :EXPECTED_NUM_FRAMES]
            elif feature.shape[1] < EXPECTED_NUM_FRAMES:
                return np.pad(feature, ((0,0), (0, EXPECTED_NUM_FRAMES - feature.shape[1])), mode='constant')
            return feature

        # Obrada značajki
        mfcc_processed = process_feature(mfcc)
        chroma_processed = process_feature(chroma)
        centroid_processed = process_feature(spectral_centroid)
        bandwidth_processed = process_feature(spectral_bandwidth)
        contrast_processed = process_feature(spectral_contrast)
        zcr_processed = process_feature(zcr)

        # Statistike
        def extract_stats(features):
            mean = np.mean(features, axis=1)
            std = np.std(features, axis=1)
            maximum = np.max(features, axis=1)
            minimum = np.min(features, axis=1)
            median = np.median(features, axis=1)
            return np.concatenate([mean, std, maximum, minimum, median])

        # Kombiniraj značajke
        features = []
        features.extend(extract_stats(mfcc_processed))
        features.extend(extract_stats(chroma_processed))
        features.extend(extract_stats(centroid_processed))
        features.extend(extract_stats(bandwidth_processed))
        features.extend(extract_stats(zcr_processed))
        features.extend(extract_stats(contrast_processed))
        features.append(tempo)

        return np.array(features).reshape(1, -1)

    except Exception as e:
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_all_models.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Učitaj mappings
        with open(os.path.join(base_dir, 'mappings.json'), 'r') as f:
            mappings = json.load(f)
        
        # Učitaj sve modele
        scaler = joblib.load(os.path.join(base_dir, 'scaler.save'))
        models = {
            "DNN": tf.keras.models.load_model(os.path.join(base_dir, 'genre_classifier.keras')),
            "RF": joblib.load(os.path.join(base_dir, 'rf_model.save')),
            "KNN": joblib.load(os.path.join(base_dir, 'knn_model.save')),
            "SVM": joblib.load(os.path.join(base_dir, 'svm_model.save'))
        }

        # Ekstrakcija značajki
        features = extract_and_process_features(audio_path)
        if features is None:
            print("Error processing audio file")
            sys.exit(1)

        # Predikcije za sve modele
        results = {}
        
        # DNN (Keras model)
        features_scaled = scaler.transform(features)
        prediction = models["DNN"].predict(features_scaled, verbose=0)
        predicted_index = int(np.argmax(prediction))
        results["DNN"] = mappings[predicted_index] if isinstance(mappings, list) else mappings[str(predicted_index)]
        
        # RF i KNN (skaliranje potrebno)
        for model_name in ["RF", "KNN"]:
            predicted_index = int(models[model_name].predict(features_scaled)[0])
            results[model_name] = mappings[predicted_index] if isinstance(mappings, list) else mappings[str(predicted_index)]
        
        # SVM (bez skaliranja - pipeline ima svoj scaler)
        predicted_index = int(models["SVM"].predict(features)[0])
        results["SVM"] = mappings[predicted_index] if isinstance(mappings, list) else mappings[str(predicted_index)]

        # Ispis rezultata u parsirljivom formatu
        print(f"DNN:{results['DNN']}")
        print(f"RF:{results['RF']}")
        print(f"KNN:{results['KNN']}")
        print(f"SVM:{results['SVM']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
