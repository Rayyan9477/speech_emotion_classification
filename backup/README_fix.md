# Speech Emotion Classification

A deep learning project to classify emotions from speech data using CNNs and MLPs.

## Project Overview

This project implements a speech emotion classification system using deep learning techniques. It can identify emotions such as happy, sad, angry, fearful, etc. from audio recordings.

## Key Features

- Data loading from RAVDESS dataset
- Feature extraction (spectrograms and MFCCs)
- CNN and MLP model architectures
- Training with early stopping and learning rate scheduling
- Evaluation with detailed metrics
- Visualization of results
- Real-time emotion prediction through a Streamlit app

## Requirements

This project requires Python 3.8+ and the following packages:
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Librosa
- Soundfile
- Streamlit
- scikit-learn

Install the required packages using:
```
pip install -r requirements.txt
```

## Usage

### Training a model

```
python main.py --model_type cnn --epochs 50
```

Options:
- `--model_type`: Type of model to train (mlp or cnn)
- `--optimize`: Whether to optimize hyperparameters
- `--epochs`: Maximum number of epochs for training
- `--batch_size`: Batch size for training

### Analyzing predictions

```
python analyze_predictions.py
```

Options:
- `--sample`: Index of specific sample to analyze
- `--model_type`: Model type (cnn or mlp)
- `--top_n`: Number of top misclassifications to analyze

### Running the web app

```
streamlit run fixed_app.py
```

### Using the new Driver Script

For a more convenient way to run all components of the system, you can use the new driver script:

```
python driver.py [options]
```

Available options:
- `--train`: Train a new model
- `--model-type {mlp,cnn}`: Type of model to train (default: cnn)
- `--optimize`: Optimize hyperparameters during training
- `--analyze`: Analyze model predictions
- `--visualize`: Generate visualizations
- `--app`: Run the Streamlit app
- `--all`: Run all steps: train, analyze, visualize, and app

Examples:
```
# Just run the app
python driver.py --app

# Train a model and then run the app
python driver.py --train --app

# Run everything
python driver.py --all
```

## Technical Notes

### Fix for TensorFlow OverflowError

This project includes a monkey patch for a TensorFlow issue that causes an OverflowError when using the `0x80000000` constant. The error typically occurs when running `argmax` on floating-point tensors:

```
OverflowError: Python int too large to convert to C long
```

The fix is implemented in `monkey_patch.py` and automatically applied when importing the main modules. This patch replaces the problematic `argmax` function with a safer implementation that avoids using `signbit` directly.

If you encounter this error in other TensorFlow projects, you can adopt a similar solution by:

1. Importing the `monkey_patch.py` file
2. Calling `monkey_patch.monkeypatch()` before any TensorFlow operations

## Model Summary

The CNN model architecture consists of:
- 2 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Dense layers with dropout
- Softmax output for emotion classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.
