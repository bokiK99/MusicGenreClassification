import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

def load_dataset(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    X_mfcc = np.array(data["mfcc"])
    X_chroma = np.array(data["chroma"])
    X_spectral_centroid = np.array(data["spectral_centroid"])
    X_spectral_bandwidth = np.array(data["spectral_bandwidth"])
    X_zcr = np.array(data["zcr"])
    X_spectral_contrast = np.array(data["spectral_contrast"])
    X_tempo = np.array(data["tempo"]).reshape(-1, 1)
    
    y = np.array(data["labels"])
    mappings = data["mappings"]
    
    return (X_mfcc, X_chroma, X_spectral_centroid, X_spectral_bandwidth, 
            X_zcr, X_spectral_contrast, X_tempo, y, mappings)

def extract_stats(features):
    mean_vals = np.mean(features, axis=2)
    std_vals = np.std(features, axis=2)
    max_vals = np.max(features, axis=2)
    min_vals = np.min(features, axis=2)
    median_vals = np.median(features, axis=2)
    return np.column_stack([mean_vals, std_vals, max_vals, min_vals, median_vals])

# File pathovi
train_path = "C:/Users/bokib/MMS_KV/train_v3.json"
val_path = "C:/Users/bokib/MMS_KV/validation_v2.json"
test_path = "C:/Users/bokib/MMS_KV/test_v2.json"

# Ucitavanje dataseta
(X_mfcc_train, X_chroma_train, X_spectral_centroid_train, X_spectral_bandwidth_train,
 X_zcr_train, X_spectral_contrast_train, X_tempo_train, y_train, mappings) = load_dataset(train_path)

(X_mfcc_val, X_chroma_val, X_spectral_centroid_val, X_spectral_bandwidth_val,
 X_zcr_val, X_spectral_contrast_val, X_tempo_val, y_val, _) = load_dataset(val_path)

(X_mfcc_test, X_chroma_test, X_spectral_centroid_test, X_spectral_bandwidth_test,
 X_zcr_test, X_spectral_contrast_test, X_tempo_test, y_test, _) = load_dataset(test_path)

# Oblici ulaznih podataka
print(f"Original MFCC shape: {X_mfcc_train.shape}")
print(f"Original Chroma shape: {X_chroma_train.shape}")
print(f"Original Spectral Centroid shape: {X_spectral_centroid_train.shape}")
print(f"Original Spectral Bandwidth shape: {X_spectral_bandwidth_train.shape}")
print(f"Original ZCR shape: {X_zcr_train.shape}")
print(f"Original Spectral Contrast shape: {X_spectral_contrast_train.shape}")
print(f"Original Tempo shape: {X_tempo_train.shape}")

# Procesuiranje znacajki
mfcc_features_train = extract_stats(X_mfcc_train)
chroma_features_train = extract_stats(X_chroma_train)
centroid_features_train = extract_stats(X_spectral_centroid_train)
bandwidth_features_train = extract_stats(X_spectral_bandwidth_train)
zcr_features_train = extract_stats(X_zcr_train)
contrast_features_train = extract_stats(X_spectral_contrast_train)

mfcc_features_val = extract_stats(X_mfcc_val)
chroma_features_val = extract_stats(X_chroma_val)
centroid_features_val = extract_stats(X_spectral_centroid_val)
bandwidth_features_val = extract_stats(X_spectral_bandwidth_val)
zcr_features_val = extract_stats(X_zcr_val)
contrast_features_val = extract_stats(X_spectral_contrast_val)

mfcc_features_test = extract_stats(X_mfcc_test)
chroma_features_test = extract_stats(X_chroma_test)
centroid_features_test = extract_stats(X_spectral_centroid_test)
bandwidth_features_test = extract_stats(X_spectral_bandwidth_test)
zcr_features_test = extract_stats(X_zcr_test)
contrast_features_test = extract_stats(X_spectral_contrast_test)

# Kombiniranje svih znacajki
X_train = np.hstack([
    mfcc_features_train,
    chroma_features_train,
    centroid_features_train,
    bandwidth_features_train,
    zcr_features_train,
    contrast_features_train,
    X_tempo_train
])

X_val = np.hstack([
    mfcc_features_val,
    chroma_features_val,
    centroid_features_val,
    bandwidth_features_val,
    zcr_features_val,
    contrast_features_val,
    X_tempo_val
])

X_test = np.hstack([
    mfcc_features_test,
    chroma_features_test,
    centroid_features_test,
    bandwidth_features_test,
    zcr_features_test,
    contrast_features_test,
    X_tempo_test
])

# Oblik znacajki
print(f"MFCC features shape: {mfcc_features_train.shape}")
print(f"Chroma features shape: {chroma_features_train.shape}")
print(f"Spectral Centroid features shape: {centroid_features_train.shape}")
print(f"Spectral Bandwidth features shape: {bandwidth_features_train.shape}")
print(f"ZCR features shape: {zcr_features_train.shape}")
print(f"Spectral Contrast features shape: {contrast_features_train.shape}")
print(f"Final combined shape: {X_train.shape}")

# Skaliranje
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Odredjivanje hiperparametara
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=123,
    n_jobs=-1
)

# Treniranje
print("\nTraining Random Forest...")
rf.fit(X_train_scaled, y_train)

# Evaluacija na validacijskom skupu
val_acc = rf.score(X_val_scaled, y_val)
print(f"\nValidation set accuracy: {val_acc:.4f}")

# Evaluacija na testnom skupu
y_pred = rf.predict(X_test_scaled)
test_acc = rf.score(X_test_scaled, y_test)
print(f"Test set accuracy: {test_acc:.4f}")

print("\nTest set performance:")
print(classification_report(y_test, y_pred, target_names=mappings))

# Matrica zabune
cm = confusion_matrix(y_test, y_pred, normalize = 'true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=mappings, yticklabels=mappings)
plt.title('Random Forest Confusion Matrix (Enhanced Features)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
plt.show()

# Vaznost znacajki
feature_importances = rf.feature_importances_
# Top 20 znacajki
indices = np.argsort(feature_importances)[-20:][::-1]

plt.figure(figsize=(12, 8))
plt.title('Top 20 Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.show()

# Spremi Random Forest model
#joblib.dump(rf, 'rf_model.save')