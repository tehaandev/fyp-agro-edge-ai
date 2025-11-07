import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
import time
import json
import datetime
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Project configuration
PROJECT_DIR = "/home/tehaan/projects/fyp-agro-edge-ai"
DATA_DIR = f"{PROJECT_DIR}/data/processed/PlantVillage_Binary"
LOGS_DIR = f"{PROJECT_DIR}/logs/disease_detection"
MODELS_DIR = f"{PROJECT_DIR}/models"

# Hyperparameters
BATCH_SIZE = 32
IMG_SIDE_LENGTH = 128
IMG_SIZE = (IMG_SIDE_LENGTH, IMG_SIDE_LENGTH)
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 2
EPOCHS = 30
LR = 1e-4
LABEL_MODE = 'binary'
CLASS_WEIGHTS = None

def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU memory growth set to True")
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected. Using CPU.")

def create_directories():
    """Create necessary directories"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Directories created: {LOGS_DIR}, {MODELS_DIR}")

def load_datasets():
    """Load training, validation, and test datasets"""
    print("\nLoading datasets...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"{DATA_DIR}/train",
        labels="inferred",
        label_mode=LABEL_MODE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val",
        labels="inferred",
        label_mode=LABEL_MODE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/test",
        labels="inferred",
        label_mode=LABEL_MODE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    print(f"Class 0 maps to: {class_names[0]}")
    print(f"Class 1 maps to: {class_names[1]}")
    
    return train_ds, val_ds, test_ds, class_names

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"), 
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.1, 0.1),
        layers.GaussianNoise(0.1),     
    ])

def prepare_datasets(train_ds, val_ds, test_ds, data_augmentation):
    """Apply augmentation and prefetching"""
    print("\nPreparing datasets with augmentation and prefetching...")
    
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=BUFFER_SIZE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    val_ds = val_ds.prefetch(buffer_size=BUFFER_SIZE)
    test_ds = test_ds.prefetch(buffer_size=BUFFER_SIZE)
    
    return train_ds, val_ds, test_ds

def build_model():
    """Build and compile the model"""
    print("\nBuilding model...")
    
    normalization_layer = layers.Rescaling(1./255)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        normalization_layer,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(IMG_SIDE_LENGTH, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Model built successfully")
    model.summary()
    
    return model

def create_callbacks():
    """Create training callbacks"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

def train_model(model, train_ds, val_ds, callbacks):
    """Train the model"""
    print("\nStarting training...")
    training_start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=CLASS_WEIGHTS,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return history, training_time

def evaluate_model(model, test_ds, class_names):
    """Evaluate the model on test set"""
    print("\nEvaluating model on test set...")
    
    test_results = model.evaluate(test_ds)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2] if len(test_results) > 2 else None
    test_recall = test_results[3] if len(test_results) > 3 else None
    
    # Get predictions for detailed metrics
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_training_log(history, training_time, test_metrics, class_names):
    """Save training log to JSON"""
    print("\nSaving training log...")
    
    training_time_minutes = training_time / 60
    training_time_hours = training_time_minutes / 60
    
    training_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "img_side_length": IMG_SIDE_LENGTH,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "buffer_size": BUFFER_SIZE,
            "label_mode": LABEL_MODE,
            "class_weights": CLASS_WEIGHTS,
            "data_augmentation": "RandomFlip + RandomRotation + RandomZoom + RandomBrightness + RandomContrast + RandomTranslation + GaussianNoise",
            "base_model": "MobileNetV2",
            "base_model_trainable": False,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "dropout_rate": 0.3,
            "dense_layer_units": IMG_SIDE_LENGTH
        },
        "dataset_info": {
            "data_dir": DATA_DIR,
            "class_names": class_names,
            "total_classes": len(class_names)
        },
        "training_time": {
            "total_seconds": float(training_time),
            "total_minutes": float(training_time_minutes),
            "total_hours": float(training_time_hours),
            "formatted": f"{int(training_time_hours):02d}h {int(training_time_minutes % 60):02d}m {int(training_time % 60):02d}s"
        },
        "results": {
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "best_val_accuracy": float(max(history.history['val_accuracy'])),
            "best_val_loss": float(min(history.history['val_loss'])),
            "epochs_trained": len(history.history['accuracy']),
            "test_accuracy": float(test_metrics['accuracy']),
            "test_loss": float(test_metrics['loss']),
            "test_f1_score": float(test_metrics['f1_score']),
            "test_precision": float(test_metrics['precision']),
            "test_recall": float(test_metrics['recall'])
        },
        "training_history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        }
    }
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{LOGS_DIR}/training_log_{timestamp_str}.json"
    
    with open(log_filename, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"Training log saved to: {log_filename}")
    
    return training_log

def save_model(model):
    """Save model in Keras and TFLite formats"""
    print("\nSaving models...")
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Keras model
    keras_model_path = f"{MODELS_DIR}/plant_disease_binary_model_{timestamp_str}.keras"
    model.save(keras_model_path)
    print(f"Keras model saved to: {keras_model_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_model_path = f"{MODELS_DIR}/plant_disease_binary_model_{timestamp_str}.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_model_path}")

def print_summary(training_log):
    """Print training summary"""
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: MobileNetV2 (frozen) + Dense({IMG_SIDE_LENGTH}) + Dense(1)")
    print(f"Training Time: {training_log['training_time']['formatted']}")
    print(f"Final Training Accuracy: {training_log['results']['final_train_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {training_log['results']['final_val_accuracy']:.4f}")
    print(f"Test Accuracy: {training_log['results']['test_accuracy']:.4f}")
    print(f"Test F1 Score: {training_log['results']['test_f1_score']:.4f}")
    print(f"Test Precision: {training_log['results']['test_precision']:.4f}")
    print(f"Test Recall: {training_log['results']['test_recall']:.4f}")
    print(f"Epochs Trained: {training_log['results']['epochs_trained']}")
    print(f"Best Validation Accuracy: {training_log['results']['best_val_accuracy']:.4f}")
    print(f"Learning Rate: {LR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*50)

def main():
    """Main training pipeline"""
    print("="*50)
    print("PLANT DISEASE DETECTION - HEADLESS TRAINING")
    print("="*50)
    
    # Setup
    setup_gpu()
    create_directories()
    
    # Load data
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    # Prepare data
    data_augmentation = create_data_augmentation()
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds, data_augmentation)
    
    # Build model
    model = build_model()
    
    # Train
    callbacks = create_callbacks()
    history, training_time = train_model(model, train_ds, val_ds, callbacks)
    
    # Evaluate
    test_metrics = evaluate_model(model, test_ds, class_names)
    
    # Save results
    training_log = save_training_log(history, training_time, test_metrics, class_names)
    save_model(model)
    
    # Print summary
    print_summary(training_log)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()