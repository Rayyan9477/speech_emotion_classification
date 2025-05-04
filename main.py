import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import logging

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model import EmotionModel
from trainer import ModelTrainer
from optimizer import GeneticOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speech_emotion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Speech Emotion Classification')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Type of model to train (mlp or cnn)')
    parser.add_argument('--optimize', action='store_true',
                        help='Whether to optimize hyperparameters using genetic algorithm')
    parser.add_argument('--population_size', type=int, default=10,
                        help='Population size for genetic algorithm')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations for genetic algorithm')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs for training')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Size of subset to use for optimization')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("Starting Speech Emotion Classification")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Hyperparameter optimization: {args.optimize}")
    
    # Step 1: Load and split dataset
    logger.info("Step 1: Loading and splitting dataset")
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    train_data, val_data, test_data = data_loader.split_dataset()
    
    # Step 2: Extract features
    logger.info("Step 2: Extracting features")
    feature_extractor = FeatureExtractor()
    
    # Extract features based on model type
    feature_type = 'mfcc' if args.model_type == 'mlp' else 'spectrogram'
    logger.info(f"Extracting {feature_type} features for {args.model_type.upper()} model")
    
    # Process training data
    train_features = feature_extractor.process_dataset(train_data, feature_type=feature_type)
    train_features = feature_extractor.normalize_features(train_features, feature_type=feature_type, fit=True)
    
    # Process validation data
    val_features = feature_extractor.process_dataset(val_data, feature_type=feature_type)
    val_features = feature_extractor.normalize_features(val_features, feature_type=feature_type, fit=False)
    
    # Process test data
    test_features = feature_extractor.process_dataset(test_data, feature_type=feature_type)
    test_features = feature_extractor.normalize_features(test_features, feature_type=feature_type, fit=False)
    
    # Get features and labels
    X_train = train_features[feature_type]
    y_train = train_features['labels']
    X_val = val_features[feature_type]
    y_val = val_features['labels']
    X_test = test_features[feature_type]
    y_test = test_features['labels']
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Step 3: Build model
    logger.info("Step 3: Building model")
    emotion_model = EmotionModel(num_classes=7)  # 7 emotion classes
    
    # Determine input shape based on model type
    if args.model_type == 'mlp':
        input_shape = (X_train.shape[1],)  # MFCC features
    else:  # CNN
        input_shape = X_train.shape[1:]  # Spectrogram features
    
    # Step 4: Optimize hyperparameters if requested
    if args.optimize:
        logger.info("Step 4: Optimizing hyperparameters")
        optimizer = GeneticOptimizer(model_type=args.model_type, num_classes=7)
        
        _, best_params, _ = optimizer.optimize(
            X_train, y_train, X_val, y_val,
            input_shape=input_shape,
            population_size=args.population_size,
            generations=args.generations,
            subset_size=args.subset_size
        )
        
        logger.info(f"Best parameters: {best_params}")
        
        # Build model with optimized parameters
        if args.model_type == 'mlp':
            model = emotion_model.build_mlp(input_shape=input_shape, params=best_params)
        else:  # CNN
            model = emotion_model.build_cnn(input_shape=input_shape, params=best_params)
    else:
        logger.info("Step 4: Using default hyperparameters")
        
        # Build model with default parameters
        if args.model_type == 'mlp':
            model = emotion_model.build_mlp(input_shape=input_shape)
        else:  # CNN
            model = emotion_model.build_cnn(input_shape=input_shape)
    
    # Step 5: Train model
    logger.info("Step 5: Training model")
    trainer = ModelTrainer(model, model_type=args.model_type)
    callbacks = emotion_model.get_callbacks(patience=5)
    
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Step 6: Evaluate model
    logger.info("Step 6: Evaluating model")
    emotion_labels = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    metrics = trainer.evaluate(X_test, y_test, emotion_labels=emotion_labels)
    
    # Print evaluation metrics
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Average precision: {metrics['precision_avg']:.4f}")
    logger.info(f"Average recall: {metrics['recall_avg']:.4f}")
    logger.info(f"Average F1-score: {metrics['f1_avg']:.4f}")
    
    # Step 7: Save model
    logger.info("Step 7: Saving model")
    model_path = f"models/{args.model_type}_emotion_model.h5"
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Speech Emotion Classification completed successfully")

if __name__ == "__main__":
    main()