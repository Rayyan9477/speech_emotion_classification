# Speech Emotion Classification System

This project implements a speech emotion classification system using neural networks and genetic algorithms for optimization. The system classifies emotions such as calm, happy, sad, angry, fearful, surprise, and disgust from speech audio using the RAVDESS dataset.

## Project Structure

```
speech_emotion_classification/
│
├── src/                       # Source code package
│   ├── data/                 # Data loading and processing
│   │   ├── __init__.py
│   │   └── data_loader.py    # Dataset loading and splitting
│   │
│   ├── features/             # Feature extraction
│   │   ├── __init__.py
│   │   └── feature_extractor.py # Audio feature extraction
│   │
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── emotion_model.py  # Model architectures
│   │   ├── trainer.py        # Model training and evaluation
│   │   └── model_manager.py  # Model management and tracking
│   │
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── monkey_patch.py   # TensorFlow fixes
│   │
│   ├── visualization/        # Visualization tools
│   │   └── __init__.py
│   │
│   └── ui/                   # User interface components
│       └── __init__.py
│
├── main.py                   # Main entry point
├── setup_package.py         # Package installation setup
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Features

- Modular and organized codebase with clear separation of concerns
- Data loading and preprocessing using the RAVDESS dataset from Hugging Face
- Feature extraction using librosa (MFCCs and spectrograms)
- Neural network models (MLP and CNN) implemented with TensorFlow/Keras
- Model training with early stopping and comprehensive evaluation metrics
- Model management system for tracking experiments and results
- User interface for easy interaction with the system
- Comprehensive logging and error handling

## Installation

### Using pip

```bash
pip install -r requirements.txt
python setup_package.py install
```

### Development Installation

For development, install with extra dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Training a Model

```bash
python -m src.main train --model-type cnn --feature-type spectrogram
```

Options:
- `--model-type`: Choose between 'mlp' or 'cnn' (default: cnn)
- `--feature-type`: Choose between 'mfcc' or 'spectrogram' (default: spectrogram)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--patience`: Early stopping patience (default: 5)

### Evaluating a Model

```bash
python -m src.main evaluate --model-id MODEL_ID --model-type MODEL_TYPE
```

### Using the UI

```bash
streamlit run src/ui/app.py
```

## Model Performance

- CNN with spectrograms: 75-85% accuracy on test set
- MLP with MFCCs: 65-75% accuracy on test set

Results may vary based on:
- Data quality and preprocessing
- Model architecture and hyperparameters
- Training duration and early stopping criteria

## Technologies Used

- TensorFlow/Keras: Deep learning framework
- librosa: Audio processing and feature extraction
- scikit-learn: Data preprocessing and metrics
- streamlit: User interface
- pandas/numpy: Data handling
- matplotlib/plotly: Visualization

## Development

For development:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/
```

## License

MIT License
