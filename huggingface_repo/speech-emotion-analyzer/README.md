
---
library: keras
language: en
tags:
- audio
- speech
- emotion-recognition
- keras
- tensorflow
metrics:
- accuracy
- f1
---

# Speech Emotion Analyzer Model

This is a Keras model trained for speech emotion recognition.

## Model Details

The model is a Convolutional Neural Network (CNN) trained on audio features (Mel Spectrograms) to classify speech into the following emotion categories:

angry, disgust, fear, happy, neutral, sad, surprise

## Usage

To use this model, you can load it using TensorFlow/Keras:

```python
import tensorflow as tf
from huggingface_hub import hf_hub_download

repo_id = "RayyanAhmed9477/speech-emotion-analyzer"
filename = "cnn_emotion_model.keras" # Assuming the model file is saved directly

# Download the model file
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the model
model = tf.keras.models.load_model(model_path)

# Now you can use the model for prediction
# (You'll need to implement feature extraction similar to the original app)
```

**Note:** This repository contains the Keras model file directly. For a more integrated Hugging Face experience, you might consider converting the model to a format like `TFAutoModel` if applicable, but loading the Keras file directly is the simplest approach for this model type.

## Training

The model was trained using the scripts in the associated repository.

## License

[Specify your license here]

## Contact

[Your contact information or link to the original project]
