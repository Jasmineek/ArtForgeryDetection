import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.metrics import AUC, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def load_data():
    """Load and preprocess the dataset"""
    # Load authentic images
    authentic_dir = 'data/authentic'
    authentic_images = []
    for filename in os.listdir(authentic_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(authentic_dir, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            authentic_images.append(img_array)
    
    # Load forged images
    forged_dir = 'data/forged'
    forged_images = []
    for filename in os.listdir(forged_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(forged_dir, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            forged_images.append(img_array)
    
    # Convert to numpy arrays
    X_authentic = np.array(authentic_images)
    X_forged = np.array(forged_images)
    
    # Create labels
    y_authentic = np.zeros(len(X_authentic))
    y_forged = np.ones(len(X_forged))
    
    # Combine datasets
    X = np.concatenate([X_authentic, X_forged])
    y = np.concatenate([y_authentic, y_forged])
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total images: {len(X)}")
    print(f"Authentic images: {len(X_authentic)}")
    print(f"Forged images: {len(X_forged)}")
    
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    return X_train, X_val, y_train, y_val

def create_model():
    """Create and return the model using ResNet50"""
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Create the model
    inputs = Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    
    # Add custom layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(), Precision(), Recall()]
    )
    
    return model

def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_data()
    
    # Create and compile model
    model = create_model()
    
    # Calculate class weights
    total_samples = len(y_train)
    authentic_samples = np.sum(y_train == 0)
    forged_samples = np.sum(y_train == 1)
    
    class_weights = {
        0: total_samples / (2 * authentic_samples),
        1: total_samples / (2 * forged_samples)
    }
    
    # Create data generators with strong augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=16,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=16,
        shuffle=False
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'van_gogh_detector.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Save final model in the newer Keras format
    model.save('van_gogh_detector.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Evaluate model
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred_classes)
    precision = precision_score(y_val, y_pred_classes)
    recall = recall_score(y_val, y_pred_classes)
    f1 = f1_score(y_val, y_pred_classes)
    auc = roc_auc_score(y_val, y_pred)
    
    # Save metrics
    with open('model_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'AUC: {auc:.4f}\n')
    
    print(f"\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == '__main__':
    main()