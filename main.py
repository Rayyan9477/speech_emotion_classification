#!/usr/bin/env python3
# main.py - Main driver script for the Speech Emotion Classification system

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import importlib.util
import numpy as np

# Import our monkey patch first - this will fix TensorFlow issues
import monkey_patch
monkey_patch.monkeypatch()
logger = logging.getLogger(__name__) # Initialize logger after monkey_patch
logger.info("Applied monkey patch for TensorFlow.")

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    tensorflow_available = True
    logger.info(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    tensorflow_available = False
    tensorflow_error = str(e)
    logger.error(f"TensorFlow not available: {tensorflow_error}")

# Import custom modules
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model import EmotionModel
from trainer import ModelTrainer
from optimizer import GeneticOptimizer
# setup_demo, analyze_predictions, visualize_results are imported dynamically when needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speech_emotion.log"),
        logging.StreamHandler()
    ]
)

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if tensorflow_available:
        tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TF_DETERMINISTIC_OPS can slow down training, enable if strict reproducibility is paramount
    # os.environ['TF_DETERMINISTIC_OPS'] = '1' 
    logger.info(f"Random seeds set with seed: {seed}")

def check_tensorflow_availability():
    """Checks and logs TensorFlow availability."""
    if not tensorflow_available:
        logger.warning(f"TensorFlow is not available or failed to import: {tensorflow_error}")
        print(f"Warning: TensorFlow is not available ({tensorflow_error}). Some functionalities might be limited.")
    return tensorflow_available

def perform_training(args):
    """Handles the model training process."""
    logger.info("Starting model training process...")
    set_seeds()
    check_tensorflow_availability()

    # Load data
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    if dataset is None:
        logger.error("Dataset loading failed. Aborting training.")
        return
    train_data, val_data, test_data = data_loader.split_dataset(
        train_size=args.train_split,
        val_size=args.val_split,
        test_size=args.test_split
    )
    logger.info(f"Dataset split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")

    # Extract features
    feature_extractor = FeatureExtractor()
    # Process a subset for faster hyperparameter optimization if specified
    train_subset = train_data
    if args.optimize and args.subset_size is not None and args.subset_size > 0 and args.subset_size < len(train_data):
        train_subset = train_data.sample(n=args.subset_size)
        logger.info(f"Using a subset of training data for optimization: {len(train_subset)} samples")
    
    X_train_features = feature_extractor.process_dataset(train_subset, feature_type=args.model_type)
    y_train = X_train_features.pop('labels')
    
    X_val_features = feature_extractor.process_dataset(val_data, feature_type=args.model_type)
    y_val = X_val_features.pop('labels')

    X_test_features = feature_extractor.process_dataset(test_data, feature_type=args.model_type)
    y_test = X_test_features.pop('labels')

    # Save test data (features and labels) for current split BEFORE model training decision
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    X_test_to_save = None
    if args.model_type == 'cnn':
        X_test_to_save = X_test_features['spectrogram'] 
    elif args.model_type == 'mlp':
        X_test_to_save = X_test_features['mfcc']
    else:
        logger.error(f"Unsupported model type for saving test data: {args.model_type}")
        # Potentially return or raise an error if this state is critical

    if X_test_to_save is not None:
        np.save(results_dir / f"{args.model_type}_X_test.npy", X_test_to_save)
        np.save(results_dir / f"{args.model_type}_y_test.npy", y_test)
        logger.info(f"Test data (features and labels) saved in {results_dir}/ for model type {args.model_type}.")
    
    # Model training decision
    model_save_path = Path("models") / f"{args.model_type}_emotion_model_final.keras"
    if model_save_path.exists() and not args.force_train:
        logger.info(f"Model {model_save_path} already exists and --force-train not specified. Skipping model building and training.")
        print(f"Found existing model: {model_save_path}. Use --force-train to retrain.")
        return # Training part is done (skipped)

    # Determine input shape and features based on model type
    if args.model_type == 'cnn':
        X_train = X_train_features['spectrogram']
        X_val = X_val_features['spectrogram']
        X_test = X_test_features['spectrogram']
        input_shape = X_train.shape[1:]
    elif args.model_type == 'mlp':
        X_train = X_train_features['mfcc']
        X_val = X_val_features['mfcc']
        X_test = X_test_features['mfcc']
        input_shape = (X_train.shape[1],)
    else:
        logger.error(f"Unsupported model type: {args.model_type}")
        return

    logger.info(f"Input shape for {args.model_type.upper()} model: {input_shape}")

    num_classes = len(np.unique(y_train))
    logger.info(f"Number of classes: {num_classes}")

    emotion_model = EmotionModel(num_classes=num_classes)
    
    best_params = None
    if args.optimize:
        logger.info("Starting hyperparameter optimization...")
        optimizer = GeneticOptimizer(model_type=args.model_type, num_classes=num_classes)
        _, best_params, _ = optimizer.optimize(
            X_train, y_train, X_val, y_val, input_shape,
            population_size=args.population_size,
            generations=args.generations,
            subset_size=args.subset_size # Pass subset_size to optimizer
        )
        logger.info(f"Optimization complete. Best parameters: {best_params}")

    # Build and train model
    if args.model_type == 'cnn':
        model = emotion_model.build_cnn(input_shape, params=best_params)
    else: # mlp
        model = emotion_model.build_mlp(input_shape, params=best_params)
    
    model.summary(print_fn=logger.info)

    trainer = ModelTrainer(model, model_type=args.model_type)
    callbacks = emotion_model.get_callbacks(patience=args.early_stopping_patience)
    
    trainer.train(X_train, y_train, X_val, y_val, 
                  batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)
    
    logger.info("Model training complete. Evaluating on test set...")
    metrics = trainer.evaluate(X_test, y_test, emotion_labels=data_loader.get_emotion_labels())
    logger.info(f"Test set evaluation metrics: {metrics}")

    # Save model (this line is reached only if training happened)
    trainer.save_model(str(model_save_path))
    logger.info(f"Trained model saved to {model_save_path}")

    # Save test data for later use by analysis/visualization scripts
    # This part is now redundant here as it's done before the training check
    # results_dir = Path("results")
    # results_dir.mkdir(parents=True, exist_ok=True)
    # np.save(results_dir / f"{args.model_type}_X_test.npy", X_test)
    # np.save(results_dir / f"{args.model_type}_y_test.npy", y_test)
    # logger.info(f"Test data (X_test, y_test) saved in {results_dir}/ directory for model type {args.model_type}.")


def setup_environment_main():
    """Sets up the environment, e.g., by creating demo files."""
    logger.info("Setting up environment (demo files)...")
    try:
        import setup_demo
        setup_demo.main()
        logger.info("Environment setup complete.")
    except ImportError:
        logger.error("Failed to import setup_demo.py. Skipping demo file setup.")
    except Exception as e:
        logger.error(f"Error during environment setup: {e}")

def analyze_predictions_main(args):
    """Runs the prediction analysis script."""
    logger.info("Analyzing model predictions...")
    try:
        import analyze_predictions
        # Simulate command-line arguments for analyze_predictions if needed
        sys_argv_backup = sys.argv
        sys.argv = ["analyze_predictions.py", "--model_type", args.model_type]
        # Add other arguments for analyze_predictions as needed based on its parser
        # For example, if analyze_predictions.py takes a model_path:
        # model_path_arg = Path("models") / f"{args.model_type}_emotion_model_final.keras"
        # if model_path_arg.exists():
        #    sys.argv.extend(["--model_path", str(model_path_arg)])
        analyze_predictions.main()
        sys.argv = sys_argv_backup # Restore original sys.argv
        logger.info("Prediction analysis complete.")
    except ImportError:
        logger.error("Failed to import analyze_predictions.py. Skipping analysis.")
    except Exception as e:
        logger.error(f"Error during prediction analysis: {e}")

def visualize_results_main(args):
    """Runs the results visualization script."""
    logger.info("Generating result visualizations...")
    try:
        import visualize_results
        # Simulate command-line arguments for visualize_results
        sys_argv_backup = sys.argv
        model_path = Path("models") / f"{args.model_type}_emotion_model_final.keras"
        if not model_path.exists():
            model_path = Path("models") / f"{args.model_type}_emotion_model.keras" # Fallback
        if not model_path.exists():
             model_path = Path("models") / f"{args.model_type}_emotion_model.h5" # Fallback

        if model_path.exists():
            sys.argv = ["visualize_results.py", "--model_path", str(model_path)]
            # Add other arguments for visualize_results as needed
            # e.g., sys.argv.extend(["--test_data", str(Path("results") / f"{args.model_type}_X_test.npy")])
            # e.g., sys.argv.extend(["--test_labels", str(Path("results") / f"{args.model_type}_y_test.npy")])
            visualize_results.main()
        else:
            logger.warning(f"Model file for {args.model_type} not found at {model_path} (or .keras/.h5). Skipping visualization.")
        sys.argv = sys_argv_backup # Restore original sys.argv
        logger.info("Result visualization complete.")
    except ImportError:
        logger.error("Failed to import visualize_results.py. Skipping visualization.")
    except Exception as e:
        logger.error(f"Error during result visualization: {e}")

def run_streamlit_app_main():
    """Launches the Streamlit application."""
    logger.info("Launching Streamlit application...")
    
    # Prefer fixed_app.py, then app_fixed.py
    app_file_to_run = "fixed_app.py" 
    if not os.path.exists(app_file_to_run):
        logger.warning(f"'{app_file_to_run}' not found. Trying 'app_fixed.py'.")
        app_file_to_run = "app_fixed.py"
        if not os.path.exists(app_file_to_run):
            logger.error(f"Neither 'fixed_app.py' nor 'app_fixed.py' found. Cannot launch app.")
            print(f"Error: Critical application files ('fixed_app.py' or 'app_fixed.py') were not found.")
            return

    cmd = ["streamlit", "run", app_file_to_run, "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
    try:
        process = subprocess.Popen(cmd)
        logger.info(f"Streamlit app '{app_file_to_run}' started with PID {process.pid}. Access it at http://localhost:8501")
        process.wait() # Wait for the app to close
    except FileNotFoundError:
        logger.error("Streamlit command not found. Make sure Streamlit is installed and in your PATH.")
        print("Error: Streamlit command not found. Please ensure Streamlit is installed.")
    except Exception as e:
        logger.error(f"Error launching Streamlit app: {e}")
        print(f"Error launching Streamlit app: {e}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Speech Emotion Classification System - Main Driver")
    
    # Actions
    parser.add_argument("--setup", action="store_true", help="Run environment setup (e.g., create demo files).")
    parser.add_argument("--train", action="store_true", help="Train a new model.")
    parser.add_argument("--app", action="store_true", help="Run the Streamlit application.")
    parser.add_argument("--analyze", action="store_true", help="Analyze model predictions.")
    parser.add_argument("--visualize", action="store_true", help="Generate result visualizations.")
    parser.add_argument("--all", action="store_true", help="Run setup, train, analyze, visualize, and then launch the app.")

    # Model and Training Parameters (relevant if --train or --all is used)
    parser.add_argument('--model_type', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Type of model to train (mlp or cnn). Also used by --analyze and --visualize.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--train_split', type=float, default=0.7, help='Proportion of dataset for training.')
    parser.add_argument('--val_split', type=float, default=0.15, help='Proportion of dataset for validation.')
    parser.add_argument('--test_split', type=float, default=0.15, help='Proportion of dataset for testing.')

    parser.add_argument('--force_train', action='store_true', 
                        help='Force retraining the model even if an existing one is found.')

    # Hyperparameter Optimization Parameters (relevant if --train and --optimize is used)
    parser.add_argument('--optimize', action='store_true',
                        help='Whether to optimize hyperparameters using genetic algorithm during training.')
    parser.add_argument('--population_size', type=int, default=10,
                        help='Population size for genetic algorithm.')
    parser.add_argument('--generations', type=int, default=5, # Reduced default for faster runs
                        help='Number of generations for genetic algorithm.')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Size of subset of training data to use for faster hyperparameter optimization (e.g., 500). Default is to use all training data.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Ensure necessary directories exist
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("uploads").mkdir(parents=True, exist_ok=True)
    Path("demo_files").mkdir(parents=True, exist_ok=True)

    actions_specified = args.setup or args.train or args.app or args.analyze or args.visualize or args.all

    if args.all:
        setup_environment_main()
        perform_training(args)
        analyze_predictions_main(args)
        visualize_results_main(args)
        run_streamlit_app_main()
    else:
        if args.setup:
            setup_environment_main()
        if args.train:
            perform_training(args)
        if args.analyze:
            analyze_predictions_main(args)
        if args.visualize:
            visualize_results_main(args)
        if args.app:
            run_streamlit_app_main()

    if not actions_specified: 
        print("No action specified. Use --help to see available options.")
        print("Defaulting to launching the Streamlit app...")
        run_streamlit_app_main()
    
    logger.info("Main script execution finished.")