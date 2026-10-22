import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib
import os

print("\nTrenutni radni direktorij:", os.getcwd())
print("Put do skripte:", os.path.abspath(__file__))

# Funkcija za citanje iz .json filea
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

# Funkcija za izdvajanje znacajki
def extract_stats(features):
    mean_vals = np.mean(features, axis=2)
    std_vals = np.std(features, axis=2)
    max_vals = np.max(features, axis=2)
    min_vals = np.min(features, axis=2)
    median_vals = np.median(features, axis=2)
    return np.column_stack([mean_vals, std_vals, max_vals, min_vals, median_vals])

#Definiranje izgleda mreze (slojevi, broj neurona)
def create_model(input_shape, num_classes):
    model = Sequential([
        # Input sloj
        Dense(512, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Skriveni slojevi
        Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output sloj
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile-anje, adam s promjenjenom brzinom ucenja
    opt = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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

# Isto za validacijjski skup
mfcc_features_val = extract_stats(X_mfcc_val)
chroma_features_val = extract_stats(X_chroma_val)
centroid_features_val = extract_stats(X_spectral_centroid_val)
bandwidth_features_val = extract_stats(X_spectral_bandwidth_val)
zcr_features_val = extract_stats(X_zcr_val)
contrast_features_val = extract_stats(X_spectral_contrast_val)

# Isto za testni skup
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

# Oznake u one-hot kodiranje
num_genres = len(mappings)
y_train_one_hot = to_categorical(y_train, num_genres)
y_val_one_hot = to_categorical(y_val, num_genres)
y_test_one_hot = to_categorical(y_test, num_genres)

# Kreiranje modela
input_shape = X_train_scaled.shape[1]
model = create_model(input_shape, num_genres)

# Informacije o modelu
model.summary()

# Kriteriji zaustavljanja/promjene learning rate-a
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# Treniranje
history = model.fit(
    X_train_scaled,
    y_train_one_hot,
    validation_data=(X_val_scaled, y_val_one_hot),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

#save_dir = "C:/Users/bokib/MMS_KV/KV5"  # Vaš radni direktorij

#model.save(os.path.join(save_dir, 'genre_classifier.keras'))
#joblib.dump(scaler, os.path.join(save_dir, 'scaler.save'))
#with open(os.path.join(save_dir, 'mappings.json'), 'w') as f:
    #json.dump(mappings, f)

# Evaluacija na testnom setu
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_one_hot)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predikcije
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test_one_hot, axis=1)

# Classification report
print("\nTest set performance:")
print(classification_report(y_test_labels, y_pred, target_names=mappings))

# Matrica zabune
cm = confusion_matrix(y_test_labels, y_pred, normalize = 'true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=mappings, yticklabels=mappings)
plt.title('Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('fcnn_confusion_matrix.png')
plt.show()

# Povijest treniranja
plt.figure(figsize=(12, 5))

# Graf preciznosti
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Graf gubitka
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('fcnn_training_history.png')
plt.show()




