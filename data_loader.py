import os
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging

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
        
    def load_dataset(self):
        """
        Load the RAVDESS dataset from Hugging Face.
        
        Returns:
            The loaded dataset.
        """
        try:
            logger.info("Loading RAVDESS dataset from Hugging Face...")
            self.dataset = load_dataset("jonatasgrosman/ravdess")
            logger.info(f"Dataset loaded successfully with {len(self.dataset['train'])} samples")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
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
            df = self.dataset['train'].to_pandas()
            
            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=df['labels']
            )
            
            # Second split: separate validation set from training set
            relative_val_size = val_size / (train_size + val_size)
            train_df, val_df = train_test_split(
                train_val_df, 
                test_size=relative_val_size, 
                random_state=self.random_state,
                stratify=train_val_df['labels']
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
    
    # Display a sample
    sample = train_data.iloc[0]
    print(f"\nSample data:")
    print(f"Audio path: {sample['audio']['path']}")
    print(f"Emotion label: {sample['labels']}")
    print(f"Speaker ID: {sample['speaker_id']}")
    print(f"Speaker gender: {sample['speaker_gender']}")