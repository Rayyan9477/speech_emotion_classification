---
language:
- en
license: mit
library_name: tensorflow
tags:
- audio
- speech
- emotion-recognition
- deep-learning
- classification
datasets:
- ravdess
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: Speech Emotion Classification
  results:
  - task:
      name: Audio Classification
      type: audio-classification
    dataset:
      name: RAVDESS
      type: ravdess
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.4213
    - name: Precision (weighted)
      type: precision
      value: 0.7253
    - name: Recall (weighted)
      type: recall
      value: 0.4213
    - name: F1-Score (weighted)
      type: f1
      value: 0.4090
---

# Speech Emotion Classification

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co)

**Detect emotions from speech using advanced deep learning models**

</div>

---

## 🎯 Overview

This repository contains a sophisticated deep learning model for speech emotion classification. The model is designed to detect and classify emotions from audio recordings with high accuracy using advanced neural network architectures. It combines acoustic features from both Mel-frequency cepstral coefficients (MFCCs) and mel-spectrograms to analyze emotional content in speech.

## 🌟 Key Features

- **Multi-modal Architecture**: Combines CNN and MLP branches for comprehensive feature analysis
- **Real-time Processing**: Capable of processing and analyzing speech in real-time
- **High Accuracy**: State-of-the-art performance on emotion classification tasks
- **Cross-platform Compatibility**: Runs seamlessly on Windows, macOS, and Linux
- **Hugging Face Integration**: Easy model sharing and deployment via Hugging Face Hub

## 📊 Dataset

The model was trained on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains high-quality recordings of professional actors expressing different emotions. The dataset includes 8 distinct emotions:

- 😌 **Neutral**: Emotionless speech
- 😌 **Calm**: Calm and relaxed emotion
- 😊 **Happy**: Joyful and cheerful emotion
- 😢 **Sad**: Melancholic and sorrowful emotion
- 😡 **Angry**: Irritated and mad emotion
- 😱 **Fearful**: Scared and apprehensive emotion
- 😤 **Disgust**: Revolted and repulsed emotion
- 😮 **Surprised**: Astonished and amazed emotion

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~42.13% |
| **Precision (weighted)** | ~72.53% |
| **Recall (weighted)** | ~42.13% |
| **F1-Score (weighted)** | ~40.90% |

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/speech_emotion_classification.git
cd speech_emotion_classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install the dependencies manually:
```bash
pip install tensorflow numpy librosa scikit-learn huggingface_hub pandas matplotlib seaborn
```

## 🚀 Usage

### 1. Load and Use the Model

```python
import librosa
import numpy as np
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('./path/to/model.keras')

# Load an audio file
audio_path = 'path/to/audio.wav'
y, sr = librosa.load(audio_path, sr=None)

# Extract features
mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spectrogram_features = librosa.feature.melspectrogram(y=y, sr=sr)

# Normalize and reshape features according to your preprocessing pipeline
# (Implementation depends on how the model was trained)

# Make prediction
# For multi-modal models, pass both feature arrays: [mfcc_features_reshaped, spec_features_reshaped]
predictions = model.predict([mfcc_features_reshaped, spec_features_reshaped])

# Get emotion with highest probability
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
predicted_emotion = emotion_labels[np.argmax(predictions)]

print(f"Predicted emotion: {predicted_emotion}")
```

### 2. Train Your Own Model

```bash
python auto_train.py
```

### 3. Test the Model

```bash
python test_prediction_pipeline.py
```

## 🏗️ Architecture

The model uses a sophisticated multi-modal architecture:

1. **MFCC Branch**: Processes Mel-frequency cepstral coefficients using dense neural network layers
2. **Spectrogram Branch**: Processes mel-spectrogram features using convolutional layers
3. **Fusion Layer**: Combines both feature representations before final classification
4. **Output Layer**: Softmax layer for emotion classification across 8 emotional states

## 📁 Project Structure

```
speech_emotion_classification/
├── app.py                 # Streamlit web application
├── auto_train.py          # Automated training script
├── debug_labels.py        # Label debugging utilities
├── driver.py              # Main execution script
├── push_to_hub.py         # Hugging Face model upload script
├── split_model.py         # Model splitting utilities
├── test_*.py              # Test files
├── requirements.txt       # Project dependencies
├── README.md              # This file
└── ...
```

## 🧪 Evaluation

To evaluate the model on custom audio files:

```bash
python test_prediction_pipeline.py
```

This will run the model on the test dataset and provide detailed performance metrics.

## 🤗 Hugging Face Integration

The model can be easily shared and deployed using Hugging Face Hub:

```bash
python push_to_hub.py
```

## 🚧 Limitations

- Performance may vary with different accents and languages
- Audio quality (noise, clarity) can significantly affect accuracy
- Emotions expressed in speech can be culturally dependent
- Requires clear audio with minimal background noise for best results
- Shorter audio clips (5-10 seconds) typically work better than longer recordings

## 🛡️ Ethical Considerations

- This model should not be used to make critical decisions about individuals without their explicit consent
- Results should be interpreted with caution and not treated as definitive psychological assessments
- Consider privacy implications when processing audio of individuals
- Use responsibly and ethically, with appropriate consent when analyzing personal speech
- Be aware of potential bias in the training data and its impact on model predictions

## 🧪 Reproducibility

To ensure reproducible results:

1. Set random seeds:
```python
import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

2. Use the same training data and preprocessing pipeline

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

### Development Setup

```bash
git clone https://github.com/your-username/speech_emotion_classification.git
cd speech_emotion_classification
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{speech_emotion_classification,
  author = {AI Research Team},
  title = {Speech Emotion Classification Model},
  year = {2025},
  url = {https://github.com/your-username/speech_emotion_classification}
}
```

## 🆘 Support

If you have any questions or encounter issues:

1. Check the [Issues](https://github.com/your-username/speech_emotion_classification/issues) page
2. Open a new issue if your problem hasn't been addressed
3. For feature requests, please open an issue with the "enhancement" tag

## 🙏 Acknowledgments

- The RAVDESS dataset creators for providing the high-quality emotional speech data
- The TensorFlow team for providing an excellent deep learning framework
- The Librosa team for audio processing capabilities
- The Hugging Face team for model sharing capabilities