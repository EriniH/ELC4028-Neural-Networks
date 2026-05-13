# Assignment 3: Neural Networks and Deep Learning

This repository contains the solutions for **Assignment 3**, which covers various advanced Deep Learning and Natural Language Processing topics, including Generative Models (VAE, GAN), Attention Mechanisms, Information Retrieval, and Large Language Model (LLM) Benchmarking.

## Problems Overview

* **Problem 1: VAE Synthetic Data with Low-Data Stabilization**
  * Training a conditional VAE on a limited dataset of 350 real examples per digit.
  * Generating synthetic datasets and classifying them using LeNet-5 to evaluate confidence-based selection.
  * *Results show that high-confidence (>= 0.9) samples provide the best regularization benefits and effectively reduce the need for real data (outperforming pure 1000-real dataset).*

* **Problem 2: GAN Synthetic Data with Low-Data Stabilization**
  * Training a conditional GAN (cDCGAN) on the same limited dataset.
  * Evaluating GAN-generated data against baseline augmentations using LeNet-5.
  * *Results match the GAN conclusion, showing that mid-confidence GAN-generated samples yield the highest accuracy boost (93.45%), although real data still reigns supreme in the GAN case.*

* **Problem 3: Understanding the Impact of Attention Mechanisms**
  * Building CNN models with and without spatial attention mechanisms for image classification (Reduced MNIST) and speech recognition (spectrograms).
  * Comparing accuracy and training time.
  * *Attention adds consistent computational overhead. While it provides negligible benefits on simple MNIST images, it aids in semantic feature isolation on complex spectrogram audio data.*

* **Problem 4: General Information Retrieval (including Q/A)**
  * Implementing classical search (TF-IDF/BM25) and semantic search (embeddings) on an Arabic book.
  * Building a Retrieval-Augmented Generation (RAG) system using an open-source small LLM from Hugging Face.

* **Problem 5: Benchmarking Arabic NLP Tasks Across LLMs**
  * Benchmarking 5 state-of-the-art LLMs (Gemini, ChatGPT, Fanar, ALLaM, Jais) on the **Arabic Emotion Detection** task.
  * Constructing a balanced dataset of MSA, Classical Arabic, and Dialects.
  * Detailed evaluation and error analysis.

Please navigate to the specific problem directories for code, datasets, and detailed reports.
