#!.venv/bin/ python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time


PROJECT_DIR = ".."
DATA_DIR = f"{PROJECT_DIR}/data/processed/tomato_disease"
LOGS_DIR = f"{PROJECT_DIR}/logs/disease_detection_multiclass"

# Finetuning parameters
BATCH_SIZE = 8
IMG_SIDE_LENGTH = 224
IMG_SIZE = (IMG_SIDE_LENGTH, IMG_SIDE_LENGTH)
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1
EPOCHS = 30
LR = 1e-4
LABEL_MODE = 'categorical'

CLASS_WEIGHTS = None


# In[ ]:


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth set to True")
    except RuntimeError as e:
        print(e)


# In[ ]:


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


# In[ ]:


import os

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"Total classes: {num_classes}")
print("Class names:", class_names)

# Calculate class weights to handle imbalance
print("\nCalculating class weights...")
train_dir = f"{DATA_DIR}/train"
total_samples = 0
class_counts = {}

for cls in class_names:
    cls_path = os.path.join(train_dir, cls)
    # Count only files
    count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
    class_counts[cls] = count
    total_samples += count

print("Class counts:", class_counts)

# Calculate weights: n_samples / (n_classes * n_samples_j)
# This gives higher weight to minority classes
CLASS_WEIGHTS = {}
for i, cls in enumerate(class_names):
    weight = total_samples / (num_classes * class_counts[cls])
    CLASS_WEIGHTS[i] = weight

print("Computed Class Weights:", CLASS_WEIGHTS)


# In[ ]:


train_ds = train_ds.shuffle(1000).prefetch(buffer_size=BUFFER_SIZE)
val_ds = val_ds.prefetch(buffer_size=BUFFER_SIZE)
test_ds = test_ds.prefetch(buffer_size=BUFFER_SIZE)


# In[ ]:


normalization_layer = layers.Rescaling(1./255)


# In[ ]:


base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # freeze base

model = tf.keras.Sequential([
    normalization_layer,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(IMG_SIDE_LENGTH, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)) # Removed as dataset is already augmented


# In[ ]:


model.compile(optimizer=Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Define callbacks for better training control
import os

# Create models directory if it doesn't exist
os.makedirs(f"{PROJECT_DIR}/models", exist_ok=True)

callbacks = [
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


# In[ ]:


# Start timing
training_start_time = time.time()
print("Starting training...")


# In[ ]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    class_weight=CLASS_WEIGHTS,
    epochs=EPOCHS,
    callbacks=callbacks
)


# In[ ]:


# End timing
training_end_time = time.time()
print(f"Training completed in {training_end_time - training_start_time:.2f} seconds")


# In[ ]:


# Minimal improvements to existing plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='val accuracy', linewidth=2)
plt.title('Accuracy', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss', linewidth=2)
plt.plot(history.history['val_loss'], label='val loss', linewidth=2)
plt.title('Loss', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")


# In[ ]:


# Evaluate the model on the test dataset
test_results = model.evaluate(test_ds)
test_loss = test_results[0]
test_accuracy = test_results[1]

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# In[ ]:


# Load and preprocess your custom image
from keras.utils import load_img, img_to_array
import numpy as np

# Path to your custom image - Update these paths as needed
custom_image_path = f"../data/custom/image.png" 

try:
    img = load_img(custom_image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array / 255.0 # Normalization is in the model

    predictions = model.predict(img_array)

    print("Raw predictions:", predictions[0])

    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions[0])

    print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Could not load or process custom image: {e}")


# In[ ]:


# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

# Get predictions for test set
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels
y_true = []
for _, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

# Calculate metrics
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"Weighted F1 Score: {f1:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# In[ ]:


# Training Run Logger
import json
import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Calculate training time
try:
    training_time_seconds = training_end_time - training_start_time
except NameError:
    epochs_completed = len(history.history['accuracy'])
    estimated_time_per_epoch = 60 
    training_time_seconds = epochs_completed * estimated_time_per_epoch

training_time_minutes = training_time_seconds / 60
training_time_hours = training_time_minutes / 60

# Prepare training log data
training_log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "img_side_length": IMG_SIDE_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "buffer_size": BUFFER_SIZE,
        "label_mode": LABEL_MODE,
        "class_weights": str(CLASS_WEIGHTS),
        "data_augmentation": "None (Pre-augmented dataset)",
        "base_model": "MobileNetV2",
        "base_model_trainable": False,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "dropout_rate": 0.3,
        "dense_layer_units": IMG_SIDE_LENGTH
    },
    "dataset_info": {
        "data_dir": DATA_DIR,
        "class_names": class_names,
        "total_classes": len(class_names)
    },
    "results": {
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "best_val_loss": float(min(history.history['val_loss'])),
        "epochs_trained": len(history.history['accuracy']),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "test_f1_score_weighted": float(f1),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall)    
    },
    "training_history": {
        "accuracy": [float(x) for x in history.history['accuracy']],
        "val_accuracy": [float(x) for x in history.history['val_accuracy']],
        "loss": [float(x) for x in history.history['loss']],
        "val_loss": [float(x) for x in history.history['val_loss']]
    }
}

# Save to JSON file with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{LOGS_DIR}/training_log_{timestamp_str}.json"

with open(log_filename, 'w') as f:
    json.dump(training_log, f, indent=2)

print(f"Training log saved to: {log_filename}")

# Print summary
print("\n=== TRAINING SUMMARY ===")
print(f"Model: MobileNetV2 (frozen) + Dense({IMG_SIDE_LENGTH}) + Dense({num_classes})")
print(f"Training Time: {training_log['training_time']['formatted'] if 'training_time' in training_log else 'N/A'}")
print(f"Final Training Accuracy: {training_log['results']['final_train_accuracy']:.4f}")
print(f"Final Validation Accuracy: {training_log['results']['final_val_accuracy']:.4f}")
print(f"Test Accuracy: {training_log['results']['test_accuracy']:.4f}")
print(f"Test F1 Score: {training_log['results']['test_f1_score_weighted']:.4f}")


# In[ ]:


# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
model_path = f"{PROJECT_DIR}/models/tomato_disease_multiclass_model.tflite"
with open(model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted to TFLite format and saved to {model_path}")

