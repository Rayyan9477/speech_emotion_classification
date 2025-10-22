import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import logging
import os
import time

# Monkey patch is now applied in src/__init__.py

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
                'num_conv_layers': 4,  # Increased to 4 layers
                'filters': [32, 64, 128, 256],  # More filters for better capacity
                'kernel_size': (3, 3),
                'pool_size': (2, 2),
                'num_dense_layers': 3,  # Increased dense layers
                'dense_units': [512, 256, 128],  # Larger dense layers
                'dropout_rate': 0.3,
                'use_residual': False,  # Disable residual for now
                'l2_regularization': 0.0001,  # L2 regularization for weights
                'use_batch_norm': True,
                'use_global_pooling': True
            }
        
        try:
            # Ensure input shape is a tuple with 3 dimensions
            if not isinstance(input_shape, tuple):
                input_shape = tuple(input_shape)
                
            logger.info(f"Original input shape: {input_shape}")
            
            if len(input_shape) != 3:
                logger.warning(f"Expected 3D input shape (height, width, channels), got {input_shape}. Attempting to fix...")
                if len(input_shape) == 2:
                    # Assuming missing channel dimension
                    input_shape = (*input_shape, 1)
                    logger.info(f"Fixed input shape to {input_shape}")
                elif len(input_shape) == 1:
                    # Try to reshape a 1D input to a reasonable 2D + channel format
                    # Calculate dimensions for a roughly square image
                    dim = int(np.sqrt(input_shape[0]))
                    if dim * dim == input_shape[0]:
                        input_shape = (dim, dim, 1)
                    else:
                        # If perfect square isn't possible, use a rectangular shape
                        input_shape = (input_shape[0], 1, 1)
                    logger.info(f"Converted 1D input shape to {input_shape}")
                else:
                    logger.error(f"Cannot fix input shape {input_shape}. Expected 3D shape.")
                    raise ValueError(f"Invalid input shape: {input_shape}. Expected 3D shape.")
            
            # Import regularizer
            from tensorflow.keras.regularizers import l2
            
            # Use functional API instead of Sequential to handle variable input shapes
            inputs = Input(shape=input_shape)
            
            # First convolutional layer
            x = Conv2D(
                filters=params['filters'][0],
                kernel_size=params['kernel_size'],
                padding='same',
                activation='relu',
                kernel_regularizer=l2(params.get('l2_regularization', 0.0001))
            )(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=params['pool_size'])(x)
            x = Dropout(params['dropout_rate'])(x)
            
            # Additional convolutional layers with residual connections
            for i in range(1, params['num_conv_layers']):
                filters = params['filters'][i] if i < len(params['filters']) else 64
                
                # Store input for residual connection
                if params.get('use_residual', False) and i > 1:
                    res_connection = x
                
                # Convolutional block
                x = Conv2D(
                    filters=filters,
                    kernel_size=params['kernel_size'],
                    padding='same',
                    activation='relu',
                    kernel_regularizer=l2(params.get('l2_regularization', 0.0001))
                )(x)
                x = BatchNormalization()(x)
                
                # Add residual connection if enabled and dimensions match
                if params.get('use_residual', False) and i > 1 and x.shape[-1] == res_connection.shape[-1]:
                    x = tf.keras.layers.add([x, res_connection])
                    logger.info(f"Added residual connection at layer {i}")
                
                x = MaxPooling2D(pool_size=params['pool_size'])(x)
                x = Dropout(params['dropout_rate'])(x)
            
            # Global average pooling to reduce parameters
            if params.get('use_global_pooling', True):
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            else:
                from tensorflow.keras.layers import Flatten
                x = Flatten()(x)
            
            # Dense layers
            for i in range(params['num_dense_layers']):
                units = params['dense_units'][i] if i < len(params['dense_units']) else 64
                x = Dense(
                    units, 
                    activation='relu',
                    kernel_regularizer=l2(params.get('l2_regularization', 0.0001))
                )(x)
                if params.get('use_batch_norm', True):
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
            
            logger.info(f"Enhanced CNN model built with {params['num_conv_layers']} convolutional layers and {params['num_dense_layers']} dense layers")
            logger.info(f"Input shape: {input_shape}")
            model.summary(print_fn=logger.info)
            
            return model
        
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    def build_multimodal(self, mfcc_input_shape, spec_input_shape, params=None):
        """
        Build a Multi-Modal model combining MFCC and spectrogram features.
        
        Args:
            mfcc_input_shape (tuple): Shape of MFCC input (n_mfcc,).
            spec_input_shape (tuple): Shape of spectrogram input (n_mels, time_steps, 1).
            params (dict): Hyperparameters for the model.
            
        Returns:
            tensorflow.keras.models.Model: The compiled multi-modal model.
        """
        if params is None:
            params = {
                'learning_rate': 0.001,
                'mfcc_layers': [256, 128],
                'spec_conv_layers': [32, 64, 128],
                'fusion_layers': [256, 128],
                'dropout_rate': 0.3,
                'l2_regularization': 0.0001
            }
        
        # MFCC branch
        mfcc_input = Input(shape=mfcc_input_shape, name='mfcc_input')
        mfcc_x = Dense(params['mfcc_layers'][0], activation='relu',
                      kernel_regularizer=l2(params.get('l2_regularization', 0.0001)))(mfcc_input)
        mfcc_x = BatchNormalization()(mfcc_x)
        mfcc_x = Dropout(params['dropout_rate'])(mfcc_x)
        
        for units in params['mfcc_layers'][1:]:
            mfcc_x = Dense(units, activation='relu',
                          kernel_regularizer=l2(params.get('l2_regularization', 0.0001)))(mfcc_x)
            mfcc_x = BatchNormalization()(mfcc_x)
            mfcc_x = Dropout(params['dropout_rate'])(mfcc_x)
        
        # Spectrogram branch (CNN)
        spec_input = Input(shape=spec_input_shape, name='spec_input')
        spec_x = Conv2D(params['spec_conv_layers'][0], (3, 3), padding='same', activation='relu',
                       kernel_regularizer=l2(params.get('l2_regularization', 0.0001)))(spec_input)
        spec_x = BatchNormalization()(spec_x)
        spec_x = MaxPooling2D((2, 2))(spec_x)
        spec_x = Dropout(params['dropout_rate'])(spec_x)
        
        for filters in params['spec_conv_layers'][1:]:
            spec_x = Conv2D(filters, (3, 3), padding='same', activation='relu',
                           kernel_regularizer=l2(params.get('l2_regularization', 0.0001)))(spec_x)
            spec_x = BatchNormalization()(spec_x)
            spec_x = MaxPooling2D((2, 2))(spec_x)
            spec_x = Dropout(params['dropout_rate'])(spec_x)
        
        # Flatten spectrogram features
        spec_x = Flatten()(spec_x)
        
        # Concatenate both branches
        combined = tf.keras.layers.concatenate([mfcc_x, spec_x], name='fusion_concat')
        
        # Fusion layers
        fusion_x = combined
        for units in params['fusion_layers']:
            fusion_x = Dense(units, activation='relu',
                            kernel_regularizer=l2(params.get('l2_regularization', 0.0001)))(fusion_x)
            fusion_x = BatchNormalization()(fusion_x)
            fusion_x = Dropout(params['dropout_rate'])(fusion_x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(fusion_x)
        
        # Create model
        model = Model(inputs=[mfcc_input, spec_input], outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Multi-modal model built with MFCC branch ({mfcc_input_shape}) and spectrogram branch ({spec_input_shape})")
        model.summary(print_fn=logger.info)
        
        return model
        
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
        os.makedirs(log_dir, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,  # Reduce LR patience is half of early stopping patience
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='epoch',
                profile_batch=0  # No profiling for faster training
            )
        ]
        # Add a simple CSV logger for robust history tracking
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            append=True,
            separator=','
        )
        callbacks.append(csv_logger)
        logger.info(f"Callbacks prepared. Log directory: {log_dir}")
        return callbacks


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