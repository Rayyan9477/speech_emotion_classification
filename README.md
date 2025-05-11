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

## Model Management and Reuse

The system is designed to train models once and then reuse them for predictions, making the application more efficient. This is implemented through the following components:

### ModelManager

The `model_manager.py` module provides a comprehensive system for managing trained models:

- **Model Registration**: Models are automatically registered after training with metadata and performance metrics
- **Model Selection**: The UI allows users to select from available pre-trained models
- **Model Reuse**: Once trained, models are saved and can be reused for future predictions without retraining

### Training Process

```bash
# Train a new CNN model
python main.py --model_type cnn

# Train a new MLP model
python main.py --model_type mlp

# Train with hyperparameter optimization
python main.py --model_type cnn --optimize
```

### Model Selection in the UI

The application includes a dedicated "Model Management" section in the UI that allows users to:

1. View all available trained models
2. Select a model to use for predictions
3. Train new models when needed
4. View model performance metrics

### Benefits of Model Reuse

- **Faster Startup**: The application loads pre-trained models instead of retraining
- **Consistent Performance**: Using the same model ensures consistent predictions
- **Efficiency**: Avoid redundant training of models, saving computational resources
- **Multiple Models**: Maintain and compare different model architectures (CNN vs MLP)

### Model Directory Structure

```
models/
├── cnn_emotion_model.keras     # Primary CNN model (Keras format)
├── cnn_emotion_model.h5        # Backup CNN model (HDF5 format)
├── mlp_emotion_model.keras     # Primary MLP model (optional)
├── mlp_emotion_model.h5        # Backup MLP model (optional)
└── model_registry.json         # Registry with model metadata
```

## Using the Model Management UI

The speech emotion classification system includes a comprehensive model management UI that provides the following features:

### Model Selection

- Browse all available trained models with their performance metrics
- Select any trained model to use for predictions
- View model details including creation date, size, and performance metrics

### Model Comparison

- Compare multiple trained models side-by-side
- Visualize model performance using interactive charts
- Review detailed metrics across different model architectures

### Model Details

- View detailed performance metrics for each model
- Visualize model performance using radar charts
- Access evaluation reports for deeper analysis

### Training New Models

- Train new models directly from the UI
- Customize training parameters (epochs, batch size)
- Enable hyperparameter optimization

### Running the Application

To run the application with all model management features:

```bash
python run_app.py
```

This will start the Streamlit application and initialize the model management system.