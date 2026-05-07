"""
Evaluation script comparing MoE vs Dense Model
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from train import load_and_preprocess_data
from model import MixtureOfExperts


class DenseModel(Model):
    """Standard dense network for comparison"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.output_layer = layers.Dense(output_dim)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)


def train_dense_model(X_train, X_test, y_train, y_test, input_dim, num_classes):
    """Train standard dense model for comparison"""
    print("\n" + "="*50)
    print("🎯 Training Dense Model (Baseline)")
    print("="*50)
    
    model = DenseModel(input_dim, hidden_dim=128, output_dim=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        verbose=0
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    return model, history, test_acc


def plot_confusion_matrix(y_true, y_pred, categories, save_path="results/confusion_matrix.png"):
    """Plot confusion matrix"""
    os.makedirs("results", exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - MoE Model')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_comparison(moe_acc, dense_acc, save_path="results/comparison_chart.png"):
    """Plot accuracy comparison"""
    os.makedirs("results", exist_ok=True)
    
    models = ['MoE (Proposed)', 'Dense Network']
    accuracies = [moe_acc, dense_acc]
    colors = ['#2E86AB', '#A23B72']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('MoE vs Dense Network: Accuracy Comparison', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Comparison chart saved to {save_path}")


def main():
    print("🚀 Model Evaluation: MoE vs Dense Network")
    print("="*50)
    
    # Load data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Number of classes: {num_classes}")
    
    # Train MoE model
    moe_model = MixtureOfExperts(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=num_classes,
        num_experts=4,
        top_k=2
    )
    
    moe_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("\n📋 MoE Model Summary:")
    _ = moe_model(tf.zeros((1, input_dim)))
    moe_model.summary()
    
    print("\n🏃 Training MoE Model...")
    moe_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=30, batch_size=64, verbose=0)
    
    moe_loss, moe_acc = moe_model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions for confusion matrix
    y_pred_logits = moe_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_logits, axis=1)
    
    # Train Dense Model
    dense_model, dense_history, dense_acc = train_dense_model(
        X_train, X_test, y_train, y_test, input_dim, num_classes
    )
    
    # Print results
    print("\n" + "="*50)
    print("📊 COMPARISON RESULTS")
    print("="*50)
    print(f"🔵 MoE Model          - Test Accuracy: {moe_acc:.4f}")
    print(f"🟢 Dense Network      - Test Accuracy: {dense_acc:.4f}")
    print(f"📈 Improvement        - {((moe_acc - dense_acc) * 100):.2f}%")
    print("="*50)
    
    # Generate plots
    categories = ['atheism', 'christian', 'graphics', 'med', 'baseball']
    plot_confusion_matrix(y_test, y_pred, categories)
    plot_comparison(moe_acc, dense_acc)
    
    print("\n✅ Evaluation complete! Check the 'results/' folder for visualizations.")


if __name__ == "__main__":
    main()
