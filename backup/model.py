import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os
import time

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
            # Ensure input shape is a tuple with 3 dimensions
            if len(input_shape) != 3:
                logger.warning(f"Expected 3D input shape (height, width, channels), got {input_shape}. Attempting to fix...")
                if len(input_shape) == 2:
                    # Assuming missing channel dimension
                    input_shape = (*input_shape, 1)
                    logger.info(f"Fixed input shape to {input_shape}")
                else:
                    logger.error(f"Cannot fix input shape {input_shape}. Expected 3D shape.")
                    raise ValueError(f"Invalid input shape: {input_shape}. Expected 3D shape.")
            
            # Use functional API instead of Sequential to handle variable input shapes
            inputs = Input(shape=input_shape)
            
            # First convolutional layer
            x = Conv2D(
                filters=params['filters'][0],
                kernel_size=params['kernel_size'],
                padding='same',
                activation='relu'
            )(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=params['pool_size'])(x)
            x = Dropout(params['dropout_rate'])(x)
            
            # Additional convolutional layers
            for i in range(1, params['num_conv_layers']):
                filters = params['filters'][i] if i < len(params['filters']) else 64
                x = Conv2D(
                    filters=filters,
                    kernel_size=params['kernel_size'],
                    padding='same',
                    activation='relu'
                )(x)
                x = BatchNormalization()(x)
                x = MaxPooling2D(pool_size=params['pool_size'])(x)
                x = Dropout(params['dropout_rate'])(x)
            
            # Flatten layer - this handles the variable input shape
            x = Flatten()(x)
            
            # Dense layers
            for i in range(params['num_dense_layers']):
                units = params['dense_units'][i] if i < len(params['dense_units']) else 64
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(params['dropout_rate'])(x)
            
            # Output layer
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model with error handling for optimizer
            try:
                optimizer = Adam(learning_rate=params['learning_rate'])
            except:
                # For older TensorFlow versions
                optimizer = Adam(lr=params['learning_rate'])
                
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"CNN model built with {params['num_conv_layers']} convolutional layers and {params['num_dense_layers']} dense layers")
            logger.info(f"Input shape: {input_shape}")
            model.summary(print_fn=logger.info)
            
            return model
        
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    def get_callbacks(self, patience=5, log_dir='logs'):
        """
        Get callbacks for model training.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            log_dir (str): Directory to save TensorBoard logs.
            
        Returns:
            list: List of callbacks.
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log directory for each run
        run_id = time.strftime('run_%Y%m%d_%H%M%S')
        log_dir = os.path.join(log_dir, run_id)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='epoch',
                profile_batch=0  # No profiling for faster training
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
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