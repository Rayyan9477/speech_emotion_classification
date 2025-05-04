import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionModel:
    """
    Class for creating neural network models for speech emotion classification.
    """
    def __init__(self, num_classes=7):
        """
        Initialize the EmotionModel with the number of emotion classes.
        
        Args:
            num_classes (int): Number of emotion classes to predict.
        """
        self.num_classes = num_classes
    
    def build_mlp(self, input_shape, params=None):
        """
        Build a Multi-Layer Perceptron (MLP) model for MFCC features.
        
        Args:
            input_shape (tuple): Shape of the input data.
            params (dict): Hyperparameters for the model.
            
        Returns:
            tensorflow.keras.models.Model: The compiled MLP model.
        """
        if params is None:
            params = {
                'learning_rate': 0.001,
                'num_layers': 2,
                'units': [128, 64],
                'dropout_rate': 0.3
            }
        
        try:
            model = Sequential()
            
            # Input layer
            model.add(Dense(params['units'][0], input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(params['dropout_rate']))
            
            # Hidden layers
            for i in range(1, params['num_layers']):
                units = params['units'][i] if i < len(params['units']) else 64
                model.add(Dense(units))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(params['dropout_rate']))
            
            # Output layer
            model.add(Dense(self.num_classes, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"MLP model built with {params['num_layers']} layers")
            model.summary(print_fn=logger.info)
            
            return model
        
        except Exception as e:
            logger.error(f"Error building MLP model: {e}")
            raise
    
    def build_cnn(self, input_shape, params=None):
        """
        Build a Convolutional Neural Network (CNN) model for spectrogram features.
        
        Args:
            input_shape (tuple): Shape of the input data (n_mels, time_steps, 1).
            params (dict): Hyperparameters for the model.
            
        Returns:
            tensorflow.keras.models.Model: The compiled CNN model.
        """
        if params is None:
            params = {
                'learning_rate': 0.001,
                'num_conv_layers': 2,
                'filters': [32, 64],
                'kernel_size': (3, 3),
                'pool_size': (2, 2),
                'num_dense_layers': 2,
                'dense_units': [128, 64],
                'dropout_rate': 0.3
            }
        
        try:
            model = Sequential()
            
            # First convolutional layer
            model.add(Conv2D(
                filters=params['filters'][0],
                kernel_size=params['kernel_size'],
                padding='same',
                input_shape=input_shape,
                activation='relu'
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=params['pool_size']))
            model.add(Dropout(params['dropout_rate']))
            
            # Additional convolutional layers
            for i in range(1, params['num_conv_layers']):
                filters = params['filters'][i] if i < len(params['filters']) else 64
                model.add(Conv2D(
                    filters=filters,
                    kernel_size=params['kernel_size'],
                    padding='same',
                    activation='relu'
                ))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=params['pool_size']))
                model.add(Dropout(params['dropout_rate']))
            
            # Flatten layer
            model.add(Flatten())
            
            # Dense layers
            for i in range(params['num_dense_layers']):
                units = params['dense_units'][i] if i < len(params['dense_units']) else 64
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(params['dropout_rate']))
            
            # Output layer
            model.add(Dense(self.num_classes, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"CNN model built with {params['num_conv_layers']} convolutional layers and {params['num_dense_layers']} dense layers")
            model.summary(print_fn=logger.info)
            
            return model
        
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    def get_callbacks(self, patience=5):
        """
        Get callbacks for model training.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            
        Returns:
            list: List of callbacks.
        """
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        ]


if __name__ == "__main__":
    # Example usage
    emotion_model = EmotionModel(num_classes=7)
    
    # Build MLP model for MFCC features
    mlp_model = emotion_model.build_mlp(input_shape=(13,))  # 13 MFCC coefficients
    
    # Build CNN model for spectrogram features
    cnn_model = emotion_model.build_cnn(input_shape=(128, 100, 1))  # (n_mels, time_steps, channels)
    
    # Get callbacks
    callbacks = emotion_model.get_callbacks()
    print(f"Callbacks: {callbacks}")