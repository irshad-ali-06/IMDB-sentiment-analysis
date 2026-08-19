# 🎬 IMDB Movie Sentiment Analysis using PyTorch RNN

Welcome to this beginner-friendly guide for the IMDB Movie Sentiment Analysis project! This repository contains a Jupyter Notebook that builds a Recurrent Neural Network (RNN) using PyTorch to classify movie reviews as either **Positive** or **Negative**.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Details](#-dataset-details)
3. [Installation for Apple MacBook Air M5](#-installation-for-apple-macbook-air-m5)
4. [Project Workflow](#-project-workflow)
5. [Model Architecture](#-model-architecture)
6. [Results](#-results)
7. [How to Run](#-how-to-run)

---

## 🔎 Project Overview
This project takes raw text reviews from the IMDB dataset, cleans the text, converts it into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency), and feeds it into a Recurrent Neural Network (RNN) built with PyTorch to predict the sentiment. 

---

## 📊 Dataset Details
The model is trained on the classic **IMDB Dataset**, which contains 50,000 movie reviews.

| Column Name | Description | Example |
| :--- | :--- | :--- |
| **review** | The text of the movie review | *"A wonderful little production..."* |
| **sentiment** | The target label (Positive/Negative) | *positive* |

*(Note: Duplicate entries are removed during preprocessing, resulting in 49,582 unique reviews).*

---

## 💻 Installation for Apple MacBook Air M5
Since you are using an **Apple MacBook Air M5**, your machine uses Apple Silicon (ARM architecture). PyTorch supports hardware acceleration on Apple Silicon via **MPS** (Metal Performance Shaders). 

Here is the step-by-step installation process optimized for your M5 Mac:

### 1. Set up a Virtual Environment
It is highly recommended to use `miniforge` (Conda for Apple Silicon) or Python's native `venv` to avoid system conflicts.
```bash
# Using venv
python3 -m venv sentiment_env
source sentiment_env/bin/activate
```

### 2. Install Required Libraries
Run the following command to install the necessary packages. The PyTorch installation will automatically detect your Apple Silicon architecture.
```bash
pip install pandas numpy scikit-learn nltk torch torchvision torchaudio
```

### 3. Download NLTK Data
Before running the notebook, you need to download specific Natural Language Toolkit (NLTK) data for text processing. You can run this in your python console or notebook:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ⚙️ Project Workflow

Here is the step-by-step breakdown of how the data flows from raw text to predictions:

*   **1. Data Loading:** Read the `IMDB Dataset.csv` using Pandas.
*   **2. Data Cleaning:**
    *   🔠 Convert all text to lowercase.
    *   🔗 Remove URLs and HTML tags.
    *   ✂️ Remove punctuation.
    *   🛑 Remove Stopwords (common words like "the", "and", "is" that don't add much meaning).
    *   🌱 **Stemming:** Reduce words to their root form (e.g., "running" becomes "run").
*   **3. Feature Engineering:**
    *   Encode `sentiment` labels: `positive` ➡️ `1`, `negative` ➡️ `0`.
    *   **Vectorization:** Apply `TfidfVectorizer` (capped at top 5000 features) to convert text into numbers.
*   **4. Data Loaders:** Split the data (80% Train, 20% Test) and wrap it in PyTorch `DataLoader` for batch processing (Batch Size = 64).
*   **5. Model Training:** Train the RNN model using Binary Cross Entropy Loss and the Adam optimizer.

---

## 🧠 Model Architecture

The neural network is a simple but effective custom PyTorch `nn.Module`. 

| Layer Type | Description | Input Dimensions | Output Dimensions |
| :--- | :--- | :--- | :--- |
| **Input Layer** | Accepts the TF-IDF vectorized text | `(Batch_Size, 5000)` | `(Batch_Size, 1, 5000)` |
| **RNN Layer** | Processes sequential features to find patterns | `5000` | `Hidden_Size = 128` |
| **Fully Connected (Linear)** | Maps the RNN hidden states to a single output value | `128` | `1` |
| **Activation** | Sigmoid function to output a probability (0 to 1) | `1` | `1` |

---

## 🏆 Results
After training for **10 Epochs**, the model evaluates its performance on the unseen test dataset.

*   **Final Accuracy:** **~85.63%**

---

## 🚀 How to Run
1. Clone this repository or download the notebook file.
2. Ensure you have the `IMDB Dataset.csv` in the same directory as the notebook.
3. Install the dependencies following the **M5 Installation** steps above.
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run all cells from top to bottom!

---
*Happy Coding! 🍏💻*
