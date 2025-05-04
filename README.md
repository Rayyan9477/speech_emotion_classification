# Speech Emotion Classification System

This project implements a speech emotion classification system using neural networks and genetic algorithms for optimization. The system classifies emotions such as calm, happy, sad, angry, fearful, surprise, and disgust from speech audio using the RAVDESS dataset.

## Project Structure

```
speech_emotion_classification/
│
├── data_loader.py       # Loads and splits the RAVDESS dataset
├── feature_extractor.py # Extracts MFCCs and spectrograms from audio
├── model.py             # Defines MLP and CNN architectures
├── trainer.py           # Manages model training and evaluation
├── optimizer.py         # Implements genetic algorithms for hyperparameter tuning
├── main.py              # Integrates all components
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Features

- Data loading and preprocessing using the RAVDESS dataset from Hugging Face
- Feature extraction using librosa (MFCCs and spectrograms)
- Neural network models (MLP and CNN) implemented with TensorFlow/Keras
- Model training with early stopping and comprehensive evaluation metrics
- Hyperparameter optimization using genetic algorithms via DEAP
- Modular and well-documented codebase

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Expected Performance

CNNs with spectrograms typically achieve 70-90% accuracy on the RAVDESS dataset, while MLPs may perform slightly worse due to simpler feature inputs.

## Technologies Used

- TensorFlow/Keras: For building and training neural networks
- scikit-learn: For preprocessing and evaluation metrics
- librosa: For extracting audio features
- DEAP: For genetic algorithms to optimize hyperparameters
- datasets (Hugging Face): For loading the RAVDESS dataset