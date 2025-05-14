"""Configuration for the speech emotion recognition system using dataclasses."""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class PathConfig:
    """Path configurations"""
    root_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    results_dir: str = "results"
    demo_files_dir: str = "demo_files"
    uploads_dir: str = "uploads"

    def __post_init__(self):
        """Convert all paths to absolute paths"""
        for field_name, path in self.__dict__.items():
            if field_name != "root_dir" and path:
                setattr(self, field_name, str(Path(self.root_dir) / path))

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    duration: float = 5.0
    hop_length: int = 512
    n_fft: int = 2048
    feature_types: List[str] = field(default_factory=lambda: ["mel_spectrogram", "mfcc", "raw_waveform"])

@dataclass
class FeatureParameters:
    """Parameters for feature extraction"""
    n_mels: int = 128
    fmax: int = 8000
    power: float = 2.0
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512

@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    mel_spectrogram: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "parameters": FeatureParameters().__dict__
    })
    mfcc: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "parameters": FeatureParameters().__dict__
    })

@dataclass
class ModelArchConfig:
    """Model architecture configuration"""
    input_shape: List[Optional[int]]
    model_path: str
    backup_path: str
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    dropout_rate: float = 0.5
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

@dataclass
class CNNConfig(ModelArchConfig):
    """CNN-specific configuration"""
    conv_layers: List[int] = field(default_factory=lambda: [32, 64, 128])
    dense_layers: List[int] = field(default_factory=lambda: [256, 128])

@dataclass
class MLPConfig(ModelArchConfig):
    """MLP-specific configuration"""
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])

@dataclass
class ModelsConfig:
    """Combined models configuration"""
    cnn: CNNConfig = field(default_factory=lambda: CNNConfig(
        input_shape=[128, None, 1],
        model_path="models/cnn_emotion_model.keras",
        backup_path="models/cnn_emotion_model.h5"
    ))
    mlp: MLPConfig = field(default_factory=lambda: MLPConfig(
        input_shape=[13],
        model_path="models/mlp_emotion_model.keras",
        backup_path="models/mlp_emotion_model.h5"
    ))

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_types: List[str] = field(default_factory=lambda: ["cnn", "mlp"])
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "calm", "happy", "sad", 
        "angry", "fearful", "disgust", "surprised"
    ])
    random_seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    verbose: int = 1

@dataclass
class UIConfig:
    """UI configuration"""
    theme: str = "light"
    chart_type: str = "interactive"
    show_advanced: bool = False
    auto_record: bool = False
    show_debug: bool = False
    plot_style: str = "modern"

@dataclass
class Config:
    """Main configuration class"""
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def __post_init__(self):
        """Create necessary directories after initialization"""
        for path in vars(self.paths).values():
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure validated")

    def get_model_config(self, model_type: str) -> ModelArchConfig:
        """Get configuration for a specific model type"""
        return getattr(self.models, model_type.lower())

    def get_model_path(self, model_type: str, use_backup: bool = False) -> str:
        """Get path for a model"""
        model_config = self.get_model_config(model_type)
        return model_config.backup_path if use_backup else model_config.model_path

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file"""
        try:
            # Convert dataclasses to dictionaries
            config_dict = {
                'paths': vars(self.paths),
                'audio': vars(self.audio),
                'features': vars(self.features),
                'models': {
                    'cnn': vars(self.models.cnn),
                    'mlp': vars(self.models.mlp)
                },
                'training': vars(self.training),
                'ui': vars(self.ui)
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """Load configuration from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            paths_config = PathConfig(**config_dict['paths'])
            audio_config = AudioConfig(**config_dict['audio'])
            features_config = FeatureConfig(**config_dict['features'])
            
            models_config = ModelsConfig(
                cnn=CNNConfig(**config_dict['models']['cnn']),
                mlp=MLPConfig(**config_dict['models']['mlp'])
            )
            
            training_config = TrainingConfig(**config_dict['training'])
            ui_config = UIConfig(**config_dict['ui'])

            return cls(
                paths=paths_config,
                audio=audio_config,
                features=features_config,
                models=models_config,
                training=training_config,
                ui=ui_config
            )
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    @classmethod
    def get_default_config(cls) -> 'Config':
        """Get default configuration instance"""
        return cls()

# Global configuration instance
config = Config()
