import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_features(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    # Ekstrakcija osnovnih značajki (ISTO KAO U FCN)
    X_mfcc = np.array(data["mfcc"])
    X_chroma = np.array(data["chroma"])
    X_centroid = np.array(data["spectral_centroid"])
    X_bandwidth = np.array(data["spectral_bandwidth"])
    X_zcr = np.array(data["zcr"])
    X_contrast = np.array(data["spectral_contrast"])
    X_tempo = np.array(data["tempo"]).reshape(-1, 1)
    y = np.array(data["labels"])

    # Funkcija za statističke značajke (ISTA KAO U FCN)
    def extract_stats(features):
        mean_vals = np.mean(features, axis=2)
        std_vals = np.std(features, axis=2)
        max_vals = np.max(features, axis=2)
        min_vals = np.min(features, axis=2)
        median_vals = np.median(features, axis=2)
        return np.column_stack([mean_vals, std_vals, max_vals, min_vals, median_vals])

    # Procesuiraj sve značajke (ISTO KAO U FCN)
    mfcc_processed = extract_stats(X_mfcc)
    chroma_processed = extract_stats(X_chroma)
    centroid_processed = extract_stats(X_centroid)
    bandwidth_processed = extract_stats(X_bandwidth)
    zcr_processed = extract_stats(X_zcr)
    contrast_processed = extract_stats(X_contrast)

    # Kombiniraj značajke u ISTOM REDOSLIJEDU kao FCN! 
    # (Ovdje je bio problem u originalnom kodu)
    X_combined = np.hstack([
        mfcc_processed,
        chroma_processed,
        centroid_processed,
        bandwidth_processed,
        zcr_processed,          # ZCR prije Spectral Contrast (kao u FCN)
        contrast_processed,     # Spectral Contrast nakon ZCR (kao u FCN)
        X_tempo
    ])
    
    print(f"Final combined shape: {X_combined.shape}")  # Mora biti (N, 176)
    return X_combined, y

def main():
    # Konfiguracija
    TRAIN_PATH = "C:/Users/bokib/MMS_KV/train_v3.json"
    TEST_PATH = "C:/Users/bokib/MMS_KV/test_v2.json"
    MODEL_DIR = "C:/Users/bokib/MMS_KV/KV6"
    PIPELINE_PATH = os.path.join(MODEL_DIR, "svm_model.save")

    # Kreiraj direktorij za modele
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Učitaj i procesuiraj podatke
    print("\nUčitavam podatke...")
    X_train, y_train = load_and_process_features(TRAIN_PATH)
    X_test, y_test = load_and_process_features(TEST_PATH)

    # Definiraj pipeline (bez PCA da bude kompatibilno s FCN)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=50,
            kernel='rbf',
            gamma=0.001,
            class_weight='balanced',
            random_state=42,
            probability=True  # Dodano za kompatibilnost s ostalim modelima
        ))
    ])

    # Treniraj model
    print("\nTreniram SVM...")
    pipeline.fit(X_train, y_train)

    # Učitaj mappings iz istog izvora kao FCN
    with open("mappings.json", "r") as f:
        mappings = json.load(f)

    # Evaluacija
    y_pred = pipeline.predict(X_test)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred, target_names=mappings))
    print(f"Točnost: {accuracy_score(y_test, y_pred):.4f}")

    # Matrica zabune
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=mappings, yticklabels=mappings)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predviđeno')
    plt.ylabel('Stvarno')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix_79posto.png')
    plt.show()

    # Spremi pipeline
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"\nModel spremljen kao: {PIPELINE_PATH}")

if __name__ == "__main__":
    main()

