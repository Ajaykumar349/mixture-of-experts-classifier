# Mixture of Experts (MoE) Classifier

## 📌 Overview

This project implements a **Sparse Mixture of Experts (MoE)** architecture for text classification using TensorFlow. MoE improves model efficiency by using multiple "expert" networks and a router that selectively activates only the most relevant experts for each input.

## 🎯 Key Features

- **4 Expert Networks** - Each specializes in different input patterns
- **Sparse Routing (Top-K=2)** - Only 2 experts activated per input
- **Load Balancing Loss** - Prevents expert collapse (one expert dominating)
- **Comparison with Dense Baseline** - Shows efficiency gains

## 📊 Dataset

**20 Newsgroups** - 5 categories (atheism, christianity, graphics, medicine, baseball)
- 2,500+ training samples
- TF-IDF vectorization with 512 features

## 🏗️ Architecture
