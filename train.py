"""
Training script for MoE Classifier
"""

import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import os

from model import MixtureOfExperts


def load_and_preprocess_data():
    """Load 20 Newsgroups dataset and convert to features"""
    print("📂 Loading 20 Newsgroups dataset...")
    
    # Use 5 categories for simplicity (good for quick training)
    categories = ['alt.atheism', 'soc.religion.christian', 
                  'comp.graphics', 'sci.med', 'rec.sport.baseball']
    
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    
    print(f"✅ Loaded {len(newsgroups.data)} samples across {len(categories)} categories")
    
    # TF-IDF Vectorization
    print("🔄 Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=512,  # Reduced for faster training
        stop_words='english',
        max_df=0.7,
        min_df=2
    )
    
    X = vectorizer.fit_transform(newsgroups.data).toarray()
    y = newsgroups.target
    
    print(f"✅ Feature shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, vectorizer


def plot_learning_curves(history, save_path="results/learning_curves.png"):
    """Plot training history"""
    os.makedirs("results", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Learning curves saved to {save_path}")


def train_moe_model(X_train, X_test, y_train, y_test):
    """Train the MoE model"""
    print("\n" + "="*50)
    print("🎯 Training Mixture of Experts Model")
    print("="*50)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create MoE model
    model = MixtureOfExperts(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=num_classes,
        num_experts=4,
        top_k=2,
        load_balancing_weight=0.01
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n📋 Model Architecture:")
    sample_input = tf.zeros((1, input_dim))
    _ = model(sample_input)
    model.summary()
    
    # Train model
    print("\n🏃 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Test Accuracy: {test_acc:.4f}")
    print(f"📊 Test Loss: {test_loss:.4f}")
    
    # Save model
    model.save_weights('results/moe_model_weights.h5')
    print("✅ Model saved to results/moe_model_weights.h5")
    
    # Plot learning curves
    plot_learning_curves(history)
    
    return model, history, test_acc


def main():
    print("🚀 Mixture of Experts Classifier - Training Pipeline")
    print("="*50)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Train MoE model
    model, history, test_acc = train_moe_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("✅ Training Complete!")
    print(f"🎯 Final Test Accuracy: {test_acc:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
