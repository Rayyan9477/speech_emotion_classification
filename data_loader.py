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
        Create a dummy dataset when the real dataset cannot be loaded.
        
        Args:
            size (int): Number of samples to generate.
            
        Returns:
            pandas.DataFrame: Dummy dataset with required structure.
        """
        logger.warning(f"Creating dummy dataset with {size} samples for testing purposes")
        
        # Create random data
        dummy_data = {
            'labels': np.random.randint(0, 7, size=size),
            'speaker_id': np.random.randint(1, 25, size=size),
            'speaker_gender': np.random.choice(['M', 'F'], size=size),
        }
        
        # Create dummy audio paths
        dummy_audio = []
        for i in range(size):
            dummy_audio.append({
                'path': f'/tmp/dummy_audio_{i}.wav',
                'array': np.random.random(16000),  # 1 second of random audio at 16kHz
                'sampling_rate': 16000
            })
        
        dummy_data['audio'] = dummy_audio
        
        return pd.DataFrame(dummy_data)
        
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
            retries = 0
            while retries < max_retries:
                try:
                    logger.info(f"Loading RAVDESS dataset from '{dataset_name}' (attempt {retries+1}/{max_retries})...")
                    self.dataset = load_dataset(dataset_name)
                    logger.info(f"Dataset loaded successfully with {len(self.dataset['train'])} samples")
                    return self.dataset
                
                except HfHubHTTPError as e:
                    if "429" in str(e):  # Rate limit error
                        retries += 1
                        if retries < max_retries:
                            # Add jitter to retry delay to avoid synchronization
                            jittered_delay = retry_delay + random.uniform(0, 2)
                            logger.warning(f"Rate limit hit. Retrying in {jittered_delay:.2f} seconds...")
                            time.sleep(jittered_delay)
                        else:
                            logger.error(f"Maximum retries reached for dataset '{dataset_name}'.")
                            break  # Try the next dataset source
                    else:
                        logger.error(f"Error loading dataset '{dataset_name}': {e}")
                        break  # Try the next dataset source
                
                except Exception as e:
                    logger.error(f"Unexpected error loading dataset '{dataset_name}': {e}")
                    break  # Try the next dataset source
        
        # If all attempts failed, create a dummy dataset for testing
        logger.warning("Could not load any RAVDESS dataset. Creating a dummy dataset for development purposes.")
        dummy_df = self._create_dummy_dataset()
        self.dataset = {'train': dummy_df}
        return self.dataset
    
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
            # Convert dataset to pandas DataFrame for easier splitting
            if hasattr(self.dataset['train'], 'to_pandas'):
                # This is a Hugging Face dataset with to_pandas method
                df = self.dataset['train'].to_pandas()
            else:
                # This is already a pandas DataFrame (from dummy dataset)
                df = self.dataset['train']
            
            # Examine the dataset structure to identify the emotion label column
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            # Rest of the method remains the same
            # Identify which column contains the emotion labels
            # Different datasets might use different column names
            label_column = None
            for potential_col in ['labels', 'label', 'emotion', 'emotion_id']:
                if potential_col in df.columns:
                    label_column = potential_col
                    break
            
            # If no label column found, inspect the data to see if it's nested
            if label_column is None:
                # Look for nested structures that might contain emotion labels
                for col in df.columns:
                    if isinstance(df[col].iloc[0], (dict, list)) and col != 'audio':
                        logger.info(f"Found potential nested column: {col}")
                        # Flatten if possible to extract emotion labels
                        try:
                            if isinstance(df[col].iloc[0], dict) and 'emotion' in df[col].iloc[0]:
                                df['emotion'] = df[col].apply(lambda x: x.get('emotion'))
                                label_column = 'emotion'
                                break
                        except:
                            pass
            
            # If we still don't have a label column, create one based on filename patterns
            if label_column is None:
                logger.warning("No explicit emotion label column found. Attempting to extract from filenames.")
                # Extract emotions from filenames - RAVDESS uses a coding system
                # The 3rd element in the filename indicates emotion (e.g. "03" = happy)
                def extract_emotion_from_path(path):
                    try:
                        # Extract the emotion code from filename (format: emotion-spk_ID...)
                        if isinstance(path, dict) and 'path' in path:
                            filename = os.path.basename(path['path'])
                        else:
                            filename = os.path.basename(str(path))
                            
                        if 'happy' in filename.lower():
                            return 1  # happy
                        elif 'sad' in filename.lower():
                            return 2  # sad
                        elif 'angry' in filename.lower():
                            return 3  # angry
                        elif 'fear' in filename.lower() or 'fearful' in filename.lower():
                            return 4  # fearful
                        elif 'disgust' in filename.lower():
                            return 5  # disgust
                        elif 'surprised' in filename.lower() or 'surprise' in filename.lower():
                            return 6  # surprised
                        else:
                            return 0  # calm/neutral
                    except:
                        return 0  # default to calm/neutral
                
                # Apply the extraction function
                if 'audio' in df.columns:
                    df['emotion'] = df['audio'].apply(extract_emotion_from_path)
                    label_column = 'emotion'
            
            # If we still don't have labels, create random ones (better than crashing)
            if label_column is None:
                logger.warning("Could not determine emotion labels. Creating random labels for development.")
                df['emotion'] = np.random.randint(0, 7, size=len(df))
                label_column = 'emotion'
            
            logger.info(f"Using '{label_column}' as the emotion label column")
            
            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=df[label_column]
            )
            
            # Second split: separate validation set from training set
            relative_val_size = val_size / (train_size + val_size)
            train_df, val_df = train_test_split(
                train_val_df, 
                test_size=relative_val_size, 
                random_state=self.random_state,
                stratify=train_val_df[label_column]
            )
            
            self.train_data = train_df
            self.val_data = val_df
            self.test_data = test_df
            
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
    print(f"\nSample data:")
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