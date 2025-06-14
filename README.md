# LLM101: Foundational NLP Representations & Micrograd

This repository contains educational, step-by-step Jupyter notebooks that demonstrate foundational concepts in Natural Language Processing (NLP) and automatic differentiation. The goal is to provide clear, well-documented code and explanations for core representation techniques and the basics of building neural networks from scratch.

## Structure

- **micrograd/**
  - `derivative.ipynb`: Introduction to automatic differentiation and computational graphs, inspired by micrograd. Includes a minimal implementation of the Value class and visualization of computation graphs.
  - `image.png`: Visualization asset for computation graphs.

- **Representations/**
  - `BagOfWords.ipynb`: Implements the Bag of Words (BoW) model from scratch, including tokenization, vocabulary building, encoding, and visualization.
  - `word2vec.ipynb`: Step-by-step implementation of the Word2Vec skip-gram model with negative sampling, including training loop, loss computation, and similarity evaluation.

## Notebooks Overview

### 1. Bag of Words

- **Goal:** Represent text documents as fixed-length vectors based on word frequency.
- **Steps:**
  - Tokenize and preprocess a sample corpus.
  - Build a vocabulary and encode each document as a frequency vector.
  - Visualize the resulting BoW matrix as a heatmap.

### 2. Word2Vec

- **Goal:** Learn dense vector representations (embeddings) for words such that similar words are close in the embedding space.
- **Steps:**
  - Prepare skip-gram training data from a corpus.
  - Define and initialize embedding matrices.
  - Implement the training loop with negative sampling and categorical cross-entropy loss.
  - Evaluate embeddings using cosine similarity.

### 3. Micrograd (Automatic Differentiation)

- **Goal:** Illustrate the basics of automatic differentiation and computational graphs.
- **Steps:**
  - Define a simple Value class to track operations and gradients.
  - Visualize computation graphs.
  - Demonstrate forward and backward passes for simple functions.

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd LLM101
   ```

2. **Install dependencies:**
   - Python 3.7+
   - Jupyter Notebook
   - numpy, matplotlib, graphviz

   You can install requirements with:
   ```sh
   pip install numpy matplotlib graphviz
   ```

3. **Run the notebooks:**
   ```sh
   jupyter notebook
   ```
   Open any notebook in the browser and run the cells step by step.

## Educational Focus

- All code is written from scratch for clarity and learning.
- Each notebook contains detailed explanations and visualizations.
- Suitable for beginners and those seeking to understand the internals of NLP representations and autodiff.

## License

MIT License

---

**Author:** [Your