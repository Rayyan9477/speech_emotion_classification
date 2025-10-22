import os
import numpy as np
import pandas as pd
import time
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging
from huggingface_hub.utils import HfHubHTTPError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and splitting the RAVDESS dataset.
    """
    def __init__(self, random_state=42):
        """
        Initialize the DataLoader with a random state for reproducibility.
        
        Args:
            random_state (int): Seed for random number generation to ensure reproducible splits.
        """
        self.random_state = random_state
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def _create_dummy_dataset(self, size=200):
        """
        Create a balanced dummy dataset with more realistic emotion-specific audio patterns.
        
        Args:
            size (int): Number of samples to generate per class.
            
        Returns:
            pandas.DataFrame: Balanced dummy dataset with proper emotion distribution.
        """
        logger.warning(f"Creating balanced dummy dataset with {size} samples per emotion class")
        
        # Get emotion labels from config
        try:
            from src.core.config import Config
            config = Config()
            emotion_labels = config.training.emotion_labels
        except ImportError:
            emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        data = []
        labels = []
        
        # Create balanced samples for each emotion
        for emotion_id, emotion_name in enumerate(emotion_labels):
            for i in range(size):
                # Generate emotion-specific audio data with more complex patterns
                duration = 3.0  # 3 seconds
                sr = 16000
                samples = int(duration * sr)
                
                # Create emotion-specific audio patterns with more complexity
                audio = self._generate_emotion_audio(emotion_name, samples, sr)
                
                # Add some noise and normalize
                audio += np.random.normal(0, 0.05, samples)
                audio = audio / np.max(np.abs(audio)) * 0.8
                
                # Create audio dict
                audio_dict = {
                    'path': f'/tmp/{emotion_name}_{i}.wav',
                    'array': audio.astype(np.float32),
                    'sampling_rate': sr
                }
                
                data.append({
                    'audio': audio_dict,
                    'emotion': emotion_id,
                    'emotion_name': emotion_name,
                    'speaker_id': i % 10 + 1,
                    'speaker_gender': 'M' if i % 2 == 0 else 'F'
                })
                labels.append(emotion_id)
        
        df = pd.DataFrame(data)
        logger.info(f"Created balanced dataset with {len(df)} samples")
        logger.info(f"Class distribution: {pd.Series(labels).value_counts().sort_index()}")
        
        return df
    
    def _generate_emotion_audio(self, emotion_name, samples, sr):
        """
        Generate more complex audio patterns specific to each emotion.
        
        Args:
            emotion_name (str): Name of the emotion
            samples (int): Number of samples to generate
            sr (int): Sample rate
            
        Returns:
            numpy.ndarray: Generated audio data
        """
        t = np.linspace(0, samples/sr, samples)
        rng = np.random.default_rng(seed=42)  # Use modern numpy random generator
        
        if emotion_name == 'neutral':
            # Steady, balanced, moderate energy - baseline pattern
            base_freq = 220
            freq_mod = 1 + 0.08 * np.sin(2 * np.pi * 1.2 * t)  # Very slight modulation
            amplitude = 0.6 * (1 + 0.1 * np.sin(2 * np.pi * 0.8 * t))  # Steady amplitude
            audio = amplitude * np.sin(2 * np.pi * base_freq * freq_mod * t)
            # Add subtle harmonics
            audio += 0.2 * amplitude * np.sin(2 * np.pi * 2 * base_freq * freq_mod * t)
            
        elif emotion_name == 'calm':
            # Very steady, low energy, minimal variation - almost flat
            base_freq = 180
            freq_mod = 1 + 0.03 * np.sin(2 * np.pi * 0.4 * t)  # Minimal modulation
            amplitude = 0.35 * (1 + 0.05 * np.sin(2 * np.pi * 0.3 * t))  # Very low, steady
            audio = amplitude * np.sin(2 * np.pi * base_freq * freq_mod * t)
            # Very subtle harmonics
            audio += 0.1 * amplitude * np.sin(2 * np.pi * 2 * base_freq * freq_mod * t)
            
        elif emotion_name == 'happy':
            # High energy, rising pitch, cheerful with multiple harmonics
            base_freq = 330
            # Strong rising trend with fast modulation
            freq_rise = base_freq * (1 + 0.5 * (t / (samples/sr)))  # Strong rising
            freq_mod = 1 + 0.4 * np.sin(2 * np.pi * 4 * t)  # Fast modulation
            amplitude = 0.8 * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))  # High, varying amplitude
            audio = amplitude * np.sin(2 * np.pi * freq_rise * freq_mod * t)
            # Rich harmonics for brightness
            audio += 0.4 * amplitude * np.sin(2 * np.pi * 2 * freq_rise * freq_mod * t)
            audio += 0.25 * amplitude * np.sin(2 * np.pi * 3 * freq_rise * freq_mod * t)
            audio += 0.15 * amplitude * np.sin(2 * np.pi * 4 * freq_rise * freq_mod * t)
            
        elif emotion_name == 'sad':
            # Low energy, falling pitch, slow mournful modulation
            base_freq = 130
            # Strong falling trend with slow modulation
            freq_fall = base_freq * (1 - 0.4 * (t / (samples/sr)))  # Strong falling
            freq_mod = 1 + 0.15 * np.sin(2 * np.pi * 0.6 * t)  # Slow modulation
            amplitude = 0.45 * (1 + 0.15 * np.sin(2 * np.pi * 1.2 * t))  # Low amplitude
            audio = amplitude * np.sin(2 * np.pi * freq_fall * freq_mod * t)
            # Limited harmonics for somber tone
            audio += 0.2 * amplitude * np.sin(2 * np.pi * 2 * freq_fall * freq_mod * t)
            
        elif emotion_name == 'angry':
            # Very high energy, sharp irregular patterns, distortion
            base_freq = 450
            # Irregular frequency with sharp changes
            freq_irregular = base_freq * (1 + 0.6 * np.sin(2 * np.pi * 6 * t) + 
                                        0.4 * np.sin(2 * np.pi * 11 * t) +
                                        0.2 * rng.normal(0, 1, samples) * 0.3)
            amplitude = 0.95 * (1 + 0.6 * np.abs(np.sin(2 * np.pi * 7 * t)))  # Very high, irregular
            audio = amplitude * np.sin(2 * np.pi * freq_irregular * t)
            # Heavy distortion and noise
            audio = np.tanh(audio * 2.5)  # Heavy clipping
            audio += 0.15 * rng.normal(0, 1, samples) * amplitude
            
        elif emotion_name == 'fearful':
            # Irregular, high frequency, trembling with sudden changes
            base_freq = 380
            # Highly irregular with rapid changes
            freq_irregular = base_freq * (1 + 0.5 * np.sin(2 * np.pi * 8 * t) +
                                        0.4 * np.sin(2 * np.pi * 13 * t) +
                                        0.3 * rng.normal(0, 1, samples) * 0.4)
            amplitude = 0.7 * (1 + 0.4 * np.sin(2 * np.pi * 9 * t) +
                              0.3 * rng.normal(0, 1, samples) * 0.3)  # Trembling
            audio = amplitude * np.sin(2 * np.pi * freq_irregular * t)
            # Add sharp harmonics
            audio += 0.3 * amplitude * np.sin(2 * np.pi * 2.7 * freq_irregular * t)
            audio += 0.2 * amplitude * np.sin(2 * np.pi * 4.1 * freq_irregular * t)
            
        elif emotion_name == 'disgust':
            # Distorted, low-mid frequency, heavy distortion and noise
            base_freq = 90
            # Slow irregular modulation
            freq_mod = 1 + 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.2 * np.sin(2 * np.pi * 3.2 * t)
            amplitude = 0.55 * (1 + 0.25 * np.sin(2 * np.pi * 2.1 * t))
            audio = amplitude * np.sin(2 * np.pi * base_freq * freq_mod * t)
            # Very heavy distortion and noise
            audio = np.tanh(audio * 4)  # Extreme clipping
            audio += 0.12 * rng.normal(0, 1, samples) * amplitude
            # Add sub-harmonics for unpleasant tone
            audio += 0.25 * amplitude * np.sin(2 * np.pi * 0.5 * base_freq * freq_mod * t)
            
        elif emotion_name == 'surprised':
            # Sudden onset, very high frequency burst, sharp decay
            base_freq = 600
            # Exponential decay envelope with very sudden onset
            envelope = np.exp(-4 * t / (samples/sr))  # Fast decay
            envelope[:int(0.02 * samples)] = 1.0  # Extremely sudden onset
            envelope[int(0.02 * samples):int(0.08 * samples)] = np.exp(-8 * (t[int(0.02 * samples):int(0.08 * samples)] - t[int(0.02 * samples)]))
            amplitude = 0.9 * envelope * (1 + 0.5 * np.sin(2 * np.pi * 25 * t))
            audio = amplitude * np.sin(2 * np.pi * base_freq * t)
            # Sharp harmonics for piercing quality
            audio += 0.4 * amplitude * np.sin(2 * np.pi * 2 * base_freq * t)
            audio += 0.25 * amplitude * np.sin(2 * np.pi * 3 * base_freq * t)
            audio += 0.15 * amplitude * np.sin(2 * np.pi * 4 * base_freq * t)
            audio += 0.1 * amplitude * np.sin(2 * np.pi * 5 * base_freq * t)
            
        else:  # fallback
            # Default pattern
            audio = np.sin(2 * np.pi * 250 * t) * 0.3
        
        return audio
        
    def _try_load_single_dataset(self, dataset_name, max_retries=5, retry_delay=5):
        """
        Try to load a single dataset with retry logic.
        
        Args:
            dataset_name (str): Name of the dataset to load
            max_retries (int): Maximum number of retries
            retry_delay (int): Base delay between retries
            
        Returns:
            The loaded dataset or None if failed
        """
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Loading RAVDESS dataset from '{dataset_name}' (attempt {retries+1}/{max_retries})...")
                dataset = load_dataset(dataset_name)
                logger.info(f"Dataset loaded successfully with {len(dataset['train'])} samples")
                return dataset
            
            except HfHubHTTPError as e:
                if "429" in str(e):  # Rate limit error
                    retries += 1
                    if retries < max_retries:
                        jittered_delay = retry_delay + random.uniform(0, 2)
                        logger.warning(f"Rate limit hit. Retrying in {jittered_delay:.2f} seconds...")
                        time.sleep(jittered_delay)
                    else:
                        logger.error(f"Maximum retries reached for dataset '{dataset_name}'.")
                        return None
                else:
                    logger.error(f"Error loading dataset '{dataset_name}': {e}")
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error loading dataset '{dataset_name}': {e}")
                return None
        
        return None
    
    def load_dataset(self, max_retries=5, retry_delay=5, alternate_datasets=None):
        """
        Load the RAVDESS dataset from Hugging Face with retry mechanism.
        
        Args:
            max_retries (int): Maximum number of retries when encountering rate limits.
            retry_delay (int): Base delay in seconds between retries (will be randomized).
            alternate_datasets (list): List of alternative dataset names to try.
            
        Returns:
            The loaded dataset.
        """
        if alternate_datasets is None:
            alternate_datasets = [
                "Codec-SUPERB/RAVDESS",
                "lhoestq/ravdess-emotion",
                "RAVDESS"
            ]
        
        # Try loading from each dataset source
        for dataset_name in alternate_datasets:
            dataset = self._try_load_single_dataset(dataset_name, max_retries, retry_delay)
            if dataset is not None:
                self.dataset = dataset
                return self.dataset
        
        # If all attempts failed, create a dummy dataset for testing
        logger.warning("Could not load any RAVDESS dataset. Creating a dummy dataset for development purposes.")
        dummy_df = self._create_dummy_dataset()
        self.dataset = {'train': dummy_df}
        return self.dataset
    
    def _find_label_column(self, df):
        """
        Find the emotion label column in the dataset.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            str: Name of the label column
        """
        # Identify which column contains the emotion labels
        for potential_col in ['labels', 'label', 'emotion', 'emotion_id']:
            if potential_col in df.columns:
                return potential_col
        
        # Look for nested structures that might contain emotion labels
        for col in df.columns:
            if isinstance(df[col].iloc[0], (dict, list)) and col != 'audio':
                logger.info(f"Found potential nested column: {col}")
                try:
                    if isinstance(df[col].iloc[0], dict) and 'emotion' in df[col].iloc[0]:
                        df['emotion'] = df[col].apply(lambda x: x.get('emotion'))
                        return 'emotion'
                except (AttributeError, KeyError, TypeError):
                    pass
        
        return None
    
    def _extract_emotion_from_filename(self, path):
        """
        Extract emotion from filename patterns.
        
        Args:
            path: File path or dict containing path
            
        Returns:
            int: Emotion index
        """
        try:
            if isinstance(path, dict) and 'path' in path:
                filename = os.path.basename(path['path'])
            else:
                filename = os.path.basename(str(path))
                
            # Map to correct indices matching config.training.emotion_labels
            if 'happy' in filename.lower():
                return 2
            elif 'sad' in filename.lower():
                return 3
            elif 'angry' in filename.lower():
                return 4
            elif 'fear' in filename.lower() or 'fearful' in filename.lower():
                return 5
            elif 'disgust' in filename.lower():
                return 6
            elif 'surprised' in filename.lower() or 'surprise' in filename.lower():
                return 7
            elif 'calm' in filename.lower():
                return 1
            else:
                return 0
        except (AttributeError, TypeError, ValueError):
            return 0
    
    def _create_emotion_labels(self, df, rng):
        """
        Create emotion labels for the dataset.
        
        Args:
            df: pandas DataFrame
            rng: numpy random generator
            
        Returns:
            str: Name of the label column
        """
        label_column = self._find_label_column(df)
        
        if label_column is None:
            logger.warning("No explicit emotion label column found. Attempting to extract from filenames.")
            if 'audio' in df.columns:
                df['emotion'] = df['audio'].apply(self._extract_emotion_from_filename)
                label_column = 'emotion'
        
        if label_column is None:
            logger.warning("Could not determine emotion labels. Creating random labels for development.")
            df['emotion'] = rng.integers(0, 7, size=len(df))
            label_column = 'emotion'
        
        return label_column
    
    def split_dataset(self, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_size (float): Proportion of data for training.
            val_size (float): Proportion of data for validation.
            test_size (float): Proportion of data for testing.
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.dataset is None:
            logger.warning("Dataset not loaded. Loading now...")
            self.load_dataset()
        
        try:
            rng = np.random.default_rng(seed=42)
            
            # Convert to DataFrame
            if hasattr(self.dataset['train'], 'to_pandas'):
                df = self.dataset['train'].to_pandas()
            else:
                df = self.dataset['train']
            
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            # Create emotion labels
            label_column = self._create_emotion_labels(df, rng)
            logger.info(f"Using '{label_column}' as the emotion label column")
            
            # Perform stratified splits
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=self.random_state,
                stratify=df[label_column]
            )
            
            relative_val_size = val_size / (train_size + val_size)
            train_df, val_df = train_test_split(
                train_val_df, test_size=relative_val_size, 
                random_state=self.random_state, stratify=train_val_df[label_column]
            )
            
            self.train_data, self.val_data, self.test_data = train_df, val_df, test_df
            
            logger.info(f"Dataset split into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
            
            return self.train_data, self.val_data, self.test_data
        
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            raise
    
    def get_data(self):
        """
        Get the split data. If data hasn't been split yet, split it with default proportions.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.train_data is None or self.val_data is None or self.test_data is None:
            logger.warning("Data not split yet. Splitting with default proportions...")
            return self.split_dataset()
        
        return self.train_data, self.val_data, self.test_data


if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    train_data, val_data, test_data = data_loader.split_dataset()
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Get the correct label column name
    label_columns = [col for col in train_data.columns if col in ['labels', 'label', 'emotion', 'emotion_id']]
    label_column = label_columns[0] if label_columns else None
    
    # Display a sample
    sample = train_data.iloc[0]
    print("\nSample data:")
    print(f"Audio path: {sample['audio']['path'] if isinstance(sample['audio'], dict) else sample['audio']}")
    if label_column:
        print(f"Emotion label: {sample[label_column]}")
    else:
        print("Emotion label: Not available")
    
    # Print speaker information if available
    if 'speaker_id' in sample:
        print(f"Speaker ID: {sample['speaker_id']}")
    if 'speaker_gender' in sample:
        print(f"Speaker gender: {sample['speaker_gender']}")
        
    print("\nDataset processing complete! The model is ready for training.")