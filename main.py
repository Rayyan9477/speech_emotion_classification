import os
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# Import our monkey patch first - this will fix the argmax function
import monkey_patch
monkey_patch.monkeypatch()

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError as e:
    tensorflow_available = False
    tensorflow_error = str(e)
    
import matplotlib.pyplot as plt

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
    if tensorflow_available:
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
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining even if a model already exists')
    
    return parser.parse_args()

def main():
    # Check if TensorFlow is available
    if not tensorflow_available:
        logger.error(f"TensorFlow is not available. Error: {tensorflow_error}")
        logger.error("Cannot proceed with speech emotion classification without TensorFlow.")
        logger.error("Please reinstall TensorFlow with 'pip install tensorflow==2.16.1'")
        sys.exit(1)
        
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Create output directories
    import os as operating_system  # Import with a different name to avoid conflict
    operating_system.makedirs('results', exist_ok=True)
    operating_system.makedirs('results/reports', exist_ok=True)
    operating_system.makedirs('models', exist_ok=True)
    operating_system.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Speech Emotion Classification")
    logger.info(f"Model type: {args.model_type}")
    
    # Define model paths based on model type
    model_path = f"models/{args.model_type}_emotion_model.keras"
    h5_model_path = f"models/{args.model_type}_emotion_model.h5"
    
    # Check if model already exists and if we should skip training
    try:
        from model_manager import ModelManager
        model_manager = ModelManager()
        
        # Check if model exists in the registry
        existing_models = model_manager.get_models(model_type=args.model_type)
        
        if not args.force_retrain and (os.path.exists(model_path) or existing_models):
            logger.info(f"Trained model already exists for type {args.model_type}")
            logger.info("Skipping training. Use --force_retrain to train again.")
            
            # Get the latest model of this type
            latest_model = model_manager.get_latest_model(model_type=args.model_type)
            if latest_model:
                logger.info(f"Using latest {args.model_type.upper()} model: {latest_model['id']}")
                logger.info(f"Model metrics: {latest_model['metrics']}")
            else:
                logger.info(f"Using model at {model_path}")
            
            # Load the existing model to use later
        else:
            if args.force_retrain:
                logger.info("Force retrain option enabled. Training new model...")
            else:
                logger.info("No existing model found. Training new model...")
    except ImportError:
        # Fall back to simple file check if ModelManager isn't available
        if not args.force_retrain and os.path.exists(model_path):
            logger.info(f"Trained model already exists at {model_path}")
            logger.info("Skipping training. Use --force_retrain to train again.")
            
            # Load the existing model to use later
        try:
            if tensorflow_available:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Successfully loaded existing model from {model_path}")
                
                # Skip to visualization if needed, or exit
                logger.info("Model is ready for use in the UI. Run the app.py to use the trained model.")
                return
            else:
                logger.error("TensorFlow not available, cannot load model")
                return
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
            logger.info("Will proceed with training a new model")
    
    logger.info(f"Hyperparameter optimization: {args.optimize}")
    
    # Step 1: Load and split dataset
    logger.info("Step 1: Loading and splitting dataset")
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    train_data, val_data, test_data = data_loader.split_dataset()
    
    # Get the correct label column name
    label_column = None
    for col in train_data.columns:
        if col in ['labels', 'label', 'emotion', 'emotion_id']:
            label_column = col
            break
    
    if label_column is None:
        logger.warning("Label column not found in dataset. The feature extractor will attempt to detect it.")
    else:
        logger.info(f"Using '{label_column}' as the label column")
    
    # Step 2: Extract features
    logger.info("Step 2: Extracting features")
    feature_extractor = FeatureExtractor()
    
    # Extract features based on model type
    feature_type = 'mfcc' if args.model_type == 'mlp' else 'spectrogram'
    logger.info(f"Extracting {feature_type} features for {args.model_type.upper()} model")
    
    # Process training data first and get the max spectrogram length
    train_features = feature_extractor.process_dataset(train_data, feature_type=feature_type)
    
    # Use the same max spectrogram length for validation and test sets to ensure consistent shapes
    max_length = train_features.get('max_length', None)
    if max_length:
        logger.info(f"Using consistent spectrogram length of {max_length} across all data splits")
    
    # Normalize training features
    train_features = feature_extractor.normalize_features(train_features, feature_type=feature_type, fit=True)
    
    # Process validation and test data with the same max_length
    val_features = feature_extractor.process_dataset(val_data, feature_type=feature_type, max_length=max_length)
    val_features = feature_extractor.normalize_features(val_features, feature_type=feature_type, fit=False)
    
    test_features = feature_extractor.process_dataset(test_data, feature_type=feature_type, max_length=max_length)
    test_features = feature_extractor.normalize_features(test_features, feature_type=feature_type, fit=False)
    
    # Get features and labels
    X_train = train_features[feature_type]
    y_train = train_features['labels']
    X_val = val_features[feature_type]
    y_val = val_features['labels']
    X_test = test_features[feature_type]
    y_test = test_features['labels']
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Save the test features and labels for later analysis
    np.save(f'results/{args.model_type}_X_test.npy', X_test)
    np.save(f'results/{args.model_type}_y_test.npy', y_test)
    logger.info(f"Test features and labels saved to results/{args.model_type}_X_test.npy and results/{args.model_type}_y_test.npy")
    
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
    callbacks = emotion_model.get_callbacks(patience=10)
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save training history as JSON for visualization
    import json
    history_path = f'results/{args.model_type}_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    logger.info(f"Training history saved to {history_path}")
    
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
    
    # Check if model already exists and only save if it doesn't
    import os
    model_path = f"models/{args.model_type}_emotion_model.keras"
    h5_model_path = f"models/{args.model_type}_emotion_model.h5"
    
    if not os.path.exists(model_path):
        # Save in .keras format (newer format)
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        logger.info(f"Model already exists at {model_path}, skipping save")
    
    # Legacy .h5 format for compatibility
    if not os.path.exists(h5_model_path):
        trainer.save_model(h5_model_path)
        logger.info(f"Model saved to {h5_model_path}")
    else:
        logger.info(f"Model already exists at {h5_model_path}, skipping save")
    
    # Register model with ModelManager
    logger.info("Registering model with ModelManager")
    try:
        from model_manager import ModelManager
        model_manager = ModelManager()
        
        # Prepare metrics for model registration
        model_metrics = {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision_avg']),
            "recall": float(metrics['recall_avg']),
            "f1": float(metrics['f1_avg']),
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "RAVDESS",
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
        
        # Register the model
        model_id = model_manager.register_model(
            model_path=model_path,
            model_type=args.model_type,
            metrics=model_metrics,
            description=f"{args.model_type.upper()} model trained on RAVDESS dataset with accuracy {metrics['accuracy']:.4f}"
        )
        
        logger.info(f"Model registered with ID: {model_id}")
        
        # Save metrics to model registry
        model_manager.save_model_metrics(model_path, model_metrics)
        logger.info("Model metrics saved to registry")
    except Exception as e:
        logger.error(f"Error registering model with ModelManager: {e}")
        logger.error("Continuing without model registration")
    
    # Step 8: Run visualizations
    logger.info("Step 8: Generating advanced visualizations")
    
    # Import the visualizer
    from visualize_results import ResultsVisualizer
    
    # Create visualizer
    visualizer = ResultsVisualizer(
        model_path=model_path,
        results_dir='results',
        interactive=True
    )
    
    # Visualize model architecture
    visualizer.visualize_model_architecture()
    
    # Visualize confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    visualizer.visualize_confusion_matrix(y_test, y_pred_classes)
    
    # Visualize t-SNE
    visualizer.visualize_tsne(X_test, y_test)
    
    # Visualize training history
    visualizer.visualize_history(history_path)
    
    # Analyze misclassifications
    misclassified_df = visualizer.analyze_misclassifications(X_test, y_test)
    
    # Generate comprehensive HTML report
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    visualizer.generate_report(metrics, misclassified_df)
    
    logger.info("Speech Emotion Classification completed successfully")

if __name__ == "__main__":
    main()