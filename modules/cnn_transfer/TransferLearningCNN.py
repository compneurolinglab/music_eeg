import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.utils import to_categorical

# Define the original CNN architecture (Kell et al., 2018)
def build_kell2018_cnn(input_shape=(256, 256, 1), num_classes_word=589, num_classes_genre=43):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Shared layers
    x = tf.keras.layers.Conv2D(96, (9,9), strides=3, activation='relu', padding='same', name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name="pool1")(x)

    x = tf.keras.layers.Conv2D(256, (5,5), strides=2, activation='relu', padding='same', name="conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name="pool2")(x)

    x = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv3")(x)

    # Word Recognition Branch (to be fine-tuned)
    x_w = tf.keras.layers.Conv2D(1024, (3,3), strides=1, activation='relu', padding='same', name="conv4_W")(x)
    x_w = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv5_W")(x_w)
    x_w = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=2, padding='same', name="pool5_W")(x_w)

    x_w = tf.keras.layers.Flatten()(x_w)
    x_w = tf.keras.layers.Dense(1024, activation='relu', name="fc6_W")(x_w)
    output_word = tf.keras.layers.Dense(num_classes_word, activation='softmax', name="fctop_W")(x_w)

    # Genre Recognition Branch (kept unchanged)
    x_g = tf.keras.layers.Conv2D(1024, (3,3), strides=1, activation='relu', padding='same', name="conv4_G")(x)
    x_g = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv5_G")(x_g)
    x_g = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=2, padding='same', name="pool5_G")(x_g)

    x_g = tf.keras.layers.Flatten()(x_g)
    x_g = tf.keras.layers.Dense(1024, activation='relu', name="fc6_G")(x_g)
    output_genre = tf.keras.layers.Dense(num_classes_genre, activation='softmax', name="fctop_G")(x_g)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output_word, output_genre])
    return model

# Build the new model with updated word branch class count
new_num_classes_word = 63
model = build_kell2018_cnn(input_shape=(256,256,1), num_classes_word=new_num_classes_word, num_classes_genre=43)

# Load pretrained weights and assign to corresponding layers (Kell et al., 2018)
weights_early_path = "Weights/network_weights_early_layers_fixed.npy"
weights_genre_path = "Weights/network_weights_genre_branch_fixed.npy"
weights_word_path = "Weights/network_weights_word_branch_fixed.npy"

# Load weight dictionaries
weights_early = np.load(weights_early_path, allow_pickle=True).item()
weights_genre = np.load(weights_genre_path, allow_pickle=True).item()
weights_word = np.load(weights_word_path, allow_pickle=True).item()

# Combine shared and genre branch weights (these layers' shapes remain unchanged)
weights_common = {**weights_early, **weights_genre}

for layer in model.layers:
    if layer.name in weights_common:
        try:
            layer.set_weights([weights_common[layer.name]['W'], weights_common[layer.name]['b']])
            print(f"Loaded weights for {layer.name}")
        except Exception as e:
            print(f"Skipping layer {layer.name} due to shape mismatch: {e}")
            
    # Load weights for word branch layers if shape matches (fctop_W needs to be retrained)
    elif layer.name in weights_word and layer.name != "fctop_W":
        try:
            layer.set_weights([weights_word[layer.name]['W'], weights_word[layer.name]['b']])
            print(f"Loaded word branch weights for {layer.name}")
        except Exception as e:
            print(f"Skipping word branch layer {layer.name} due to shape mismatch: {e}")

# Set training strategy: freeze shared and genre layers, fine-tune only word branch
word_branch_names = {"conv4_W", "conv5_W", "pool5_W", "fc6_W", "fctop_W"}
for layer in model.layers:
    if layer.name in word_branch_names:
        layer.trainable = True
    else:
        layer.trainable = False

# Print model summary to check trainable status
model.summary()

# Load label_map and update number of word classes
label_map_path = "TrainSet/labels/wordLabel.json"
with open(label_map_path, "r", encoding="utf-8") as f:
    label2id = json.load(f)
new_num_classes_word = len(label2id)
print("Number of Chinese word classes:", new_num_classes_word)  # Should be 63

# Build model with updated word class count
model = build_kell2018_cnn(input_shape=(256,256,1), 
                           num_classes_word=new_num_classes_word, 
                           num_classes_genre=43)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss={'fctop_W': 'categorical_crossentropy', 
                    'fctop_G': 'categorical_crossentropy'},
              loss_weights={'fctop_W': 1.0, 'fctop_G': 0.0},
              metrics={'fctop_W': 'accuracy'})

# Load training data and generate labels
npy_folder = "TrainSet/cochleagrams_npy/"

X_train_list = []
y_train_list = []
for filename in os.listdir(npy_folder):
    if filename.endswith(".npy"):
        filepath = os.path.join(npy_folder, filename)
        cochleagram = np.load(filepath)
        # Expand to (256,256,1) if data is 2D
        if cochleagram.ndim == 2:
            cochleagram = np.expand_dims(cochleagram, axis=-1)
        X_train_list.append(cochleagram)
        # Extract label from filename
        base_name = os.path.splitext(filename)[0]
        if "_" in base_name:
            label = base_name.split("_")[0]
        else:
            label = base_name
        label_id = label2id[label]
        y_train_list.append(label_id)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
# One-hot encode labels, shape (num_samples, 63)
y_train = to_categorical(y_train, num_classes=new_num_classes_word)

print("Training data X_train shape:", X_train.shape)
print("Training labels y_train shape:", y_train.shape)

# Shuffle the data
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# Generate dummy labels for fctop_G (genre output), assuming 43 classes
dummy_y_genre = np.zeros((X_train.shape[0], 43))

# Fine-tune the model: provide both outputs
history = model.fit(X_train, {'fctop_W': y_train, 'fctop_G': dummy_y_genre},
                    batch_size=32,
                    epochs=3,
                    validation_split=0.1)
print("Fine-tuning completed!")

# Save the fine-tuned model
model_save_path = "fine_tuned_model.h5"
model.save(model_save_path)
print(f"Training complete. Model saved to {model_save_path}")
