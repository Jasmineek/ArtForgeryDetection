import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2L
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.metrics import AUC, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import datetime
import shutil
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

def setup_directories():
    """Create necessary directories for the project"""
    directories = ['data/train/authentic', 'data/train/forged',
                  'data/val/authentic', 'data/val/forged',
                  'data/test/authentic', 'data/test/forged',
                  'models', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_and_split_data():
    """Load and split the dataset into train, validation, and test sets"""
    # Load authentic images
    authentic_dir = 'data/authentic'
    authentic_images = []
    authentic_paths = []
    
    print("Loading authentic images...")
    for filename in os.listdir(authentic_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(authentic_dir, filename)
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                authentic_images.append(img_array)
                authentic_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    # Load forged images
    forged_dir = 'data/forged'
    forged_images = []
    forged_paths = []
    
    print("Loading forged images...")
    for filename in os.listdir(forged_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(forged_dir, filename)
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                forged_images.append(img_array)
                forged_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    X_authentic = np.array(authentic_images)
    X_forged = np.array(forged_images)
    
    # Create labels
    y_authentic = np.zeros(len(X_authentic))
    y_forged = np.ones(len(X_forged))
    
    # Combine datasets
    X = np.concatenate([X_authentic, X_forged])
    y = np.concatenate([y_authentic, y_forged])
    paths = authentic_paths + forged_paths
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total images: {len(X)}")
    print(f"Authentic images: {len(X_authentic)}")
    print(f"Forged images: {len(X_forged)}")
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X_temp, y_temp, paths_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Organize files into directories
    def copy_files_to_split(split_paths, split_labels, split_name):
        for path, label in zip(split_paths, split_labels):
            class_name = 'authentic' if label == 0 else 'forged'
            dest_dir = f'data/{split_name}/{class_name}'
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))
    
    copy_files_to_split(paths_train, y_train, 'train')
    copy_files_to_split(paths_val, y_val, 'val')
    copy_files_to_split(paths_test, y_test, 'test')
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(input_shape=(224, 224, 3)):
    """Create an improved model architecture"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Use a more powerful base model
    base_model = EfficientNetV2L(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Add more sophisticated layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Add more dense layers with stronger regularization
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)  # Increase dropout
    
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer with stronger regularization
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
    
    return Model(inputs=inputs, outputs=outputs)

def create_data_generators():
    """Create data generators with improved augmentation"""
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3],
        channel_shift_range=30,
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    
    # Validation data augmentation (only rescaling)
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    
    # Set up data generators
    train_generator = train_datagen.flow_from_directory(
        'training_data',
        target_size=(224, 224),
        batch_size=16,  # Reduced batch size for better generalization
        class_mode='binary',
        subset='training'
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        'training_data',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, valid_generator

def create_callbacks(model_name):
    """Create callbacks for training"""
    # Create logs directory with timestamp
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            min_delta=0.001
        ),
        ModelCheckpoint(
            f'models/{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label=f'Training {metric}')
            axes[idx].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            axes[idx].set_title(f'Model {metric}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric)
            axes[idx].legend()
            axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, filepath='model_metrics.txt'):
    """Save evaluation metrics to a file"""
    with open(filepath, 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value:.4f}\n')

def train_model():
    """Train the model with improved techniques"""
    # Create data generators
    train_generator, valid_generator = create_data_generators()
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create model
    model = create_model()
    
    # Use a custom learning rate schedule
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Use a more sophisticated optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.001,
        clipnorm=1.0
    )

    # Compile with additional metrics
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    # Set up callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ModelCheckpoint(
            'van_gogh_detector_improved.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Initial training
    print("Starting initial training...")
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Fine-tuning
    print("\nStarting fine-tuning...")
    # Unfreeze the top layers of the base model
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Fine-tune the model
    history_fine = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save the final model
    model.save('van_gogh_detector_final_improved.keras')
    
    # Print final metrics
    final_metrics = model.evaluate(valid_generator)
    print(f"\nFinal Validation Accuracy: {final_metrics[1]:.4f}")
    print(f"Final Validation Loss: {final_metrics[0]:.4f}")
    print(f"Final Validation AUC: {final_metrics[2]:.4f}")
    
    # Print class weights
    print("\nClass Weights:")
    for class_idx, weight in class_weight_dict.items():
        print(f"Class {class_idx}: {weight:.4f}")

if __name__ == '__main__':
    train_model() 