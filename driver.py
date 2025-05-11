#!/usr/bin/env python3
# driver.py - Main driver script for the Speech Emotion Classification system

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speech_emotion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def import_monkey_patch():
    """Import and apply the monkey patch to fix TensorFlow issues"""
    try:
        import monkey_patch
        monkey_patch.monkeypatch()
        logger.info("Applied monkey patch for TensorFlow")
        return True
    except ImportError as e:
        logger.error(f"Failed to import monkey_patch module: {e}")
        return False

def check_tensorflow():
    """Check if TensorFlow is available and properly configured"""
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU devices available: {len(gpus)}")
            for gpu in gpus:
                logger.info(f"  {gpu}")
        else:
            logger.info("No GPU devices found, using CPU")
        
        return True, tf.__version__
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        return False, str(e)

def check_required_modules():
    """Check if all required modules are available"""
    required_modules = [
        "numpy", "pandas", "sklearn", "matplotlib", "streamlit", 
        "librosa", "tensorflow", "keras", "deap"
    ]
    
    missing = []
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    
    if missing:
        logger.warning(f"Missing required modules: {', '.join(missing)}")
        return False, missing
    
    logger.info("All required modules available")
    return True, []

def check_model_files():
    """Check if model files exist and are valid"""
    model_dir = Path("models")
    if not model_dir.exists():
        logger.warning("Models directory does not exist")
        os.makedirs(model_dir, exist_ok=True)
        logger.info("Created models directory")
        return False
    
    model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
    if not model_files:
        logger.warning("No model files found in models directory")
        return False
    
    logger.info(f"Found {len(model_files)} model files")
    for model_file in model_files:
        logger.info(f"  {model_file}")
    
    return True

def setup_environment():
    """Set up the environment for the application"""
    # Ensure required directories exist
    dirs = ["uploads", "demo_files", "models", "results", "logs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Prepare demo files if needed
    if not os.path.exists("demo_files/happy_sample.wav") or \
       not os.path.exists("demo_files/angry_sample.wav") or \
       not os.path.exists("demo_files/sad_sample.wav"):
        logger.info("Setting up demo files...")
        try:
            import setup_demo
            setup_demo.main()
        except ImportError as e:
            logger.error(f"Failed to import setup_demo module: {e}")
    
    return True

def train_model(model_type="cnn", optimize=False):
    """Train a new model or optimize an existing one"""
    logger.info(f"Training new {model_type.upper()} model (optimize={optimize})")
    
    try:
        # Import main module
        from main import main as main_train
        import sys
        
        # Prepare arguments for training
        sys.argv = ["main.py", f"--model_type={model_type}"]
        if optimize:
            sys.argv.append("--optimize")
        
        # Run training
        main_train()
        return True
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    logger.info("Starting Streamlit application")
    
    # Check if app_fixed.py exists and use it instead of app.py if available
    app_file = "app_fixed.py" if os.path.exists("app_fixed.py") else "app.py"
    
    try:
        # Run Streamlit app
        cmd = ["streamlit", "run", app_file, "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
        process = subprocess.Popen(cmd)
        
        logger.info(f"Streamlit app is running with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {e}")
        return None

def analyze_predictions():
    """Run prediction analysis"""
    logger.info("Analyzing model predictions")
    
    try:
        import analyze_predictions
        analyze_predictions.main()
        return True
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")
        return False

def visualize_results():
    """Generate visualizations for model results"""
    logger.info("Generating visualization results")
    
    try:
        import visualize_results
        # Check if we have a model and test data
        model_path = os.path.join("models", "cnn_emotion_model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join("models", "cnn_emotion_model.h5")
        
        if os.path.exists(model_path):
            # Call with minimal arguments
            sys.argv = ["visualize_results.py", f"--model_path={model_path}"]
            visualize_results.main()
            return True
        else:
            logger.warning("No model file found for visualization")
            return False
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Speech Emotion Classification System")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--model-type", choices=["mlp", "cnn"], default="cnn", 
                      help="Type of model to train (default: cnn)")
    parser.add_argument("--optimize", action="store_true", 
                      help="Optimize hyperparameters during training")
    parser.add_argument("--analyze", action="store_true", 
                      help="Analyze model predictions")
    parser.add_argument("--visualize", action="store_true", 
                      help="Generate visualizations")
    parser.add_argument("--app", action="store_true", 
                      help="Run the Streamlit app")
    parser.add_argument("--all", action="store_true", 
                      help="Run all steps: train, analyze, visualize, and app")
    
    return parser.parse_args()

def main():
    """Main entry point for the driver script"""
    args = parse_arguments()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        print("No arguments provided. Use --help to see available options.")
        print("Running with --app flag to start the Streamlit application.")
        args.app = True
    
    # Apply monkey patch
    if not import_monkey_patch():
        logger.warning("Failed to apply monkey patch, may encounter TensorFlow errors")
    
    # Check environment
    tf_available, tf_version = check_tensorflow()
    if not tf_available:
        logger.warning(f"TensorFlow not available: {tf_version}")
        print(f"TensorFlow not available: {tf_version}")
        print("Some functionality may be limited.")
    
    modules_available, missing_modules = check_required_modules()
    if not modules_available:
        logger.warning(f"Missing required modules: {', '.join(missing_modules)}")
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Please install the missing modules to ensure full functionality.")
    
    # Setup environment
    setup_environment()
    
    # Check for model files
    models_exist = check_model_files()
    if not models_exist and (args.app or args.analyze or args.visualize or args.all):
        logger.warning("No models found but app, analyze, or visualize requested")
        print("No pre-trained models found. Training a default model first...")
        train_model()
    
    # Execute requested actions
    if args.train or args.all:
        train_model(model_type=args.model_type, optimize=args.optimize)
    
    if args.analyze or args.all:
        analyze_predictions()
    
    if args.visualize or args.all:
        visualize_results()
    
    # Run the Streamlit app last (it blocks until closed)
    if args.app or args.all:
        app_process = run_streamlit_app()
        if app_process:
            try:
                app_process.wait()
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping app")
                app_process.terminate()
    
    logger.info("Driver script completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
