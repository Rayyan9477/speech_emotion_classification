import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Class for extracting audio features (MFCCs and spectrograms) from audio files.
    """
    def __init__(self, n_mfcc=13, n_mels=128, hop_length=512, n_fft=2048):
        """
        Initialize the FeatureExtractor with parameters for feature extraction.
        
        Args:
            n_mfcc (int): Number of MFCC coefficients to extract.
            n_mels (int): Number of Mel bands to generate for spectrograms.
            hop_length (int): Number of samples between successive frames.
            n_fft (int): Length of the FFT window.
        """
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.scaler_mfcc = StandardScaler()
        self.scaler_spec = StandardScaler()
        
    def extract_mfcc(self, audio_data, sr):
        """
        Extract MFCC features from audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio time series.
            sr (int): Sample rate of the audio.
            
        Returns:
            numpy.ndarray: Extracted MFCC features.
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Transpose to get time as the first dimension
            mfccs = mfccs.T
            
            # Average over time for MLP input
            mfcc_features = np.mean(mfccs, axis=0)
            
            return mfcc_features
        
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            raise
    
    def extract_spectrogram(self, audio_data, sr):
        """
        Extract Mel spectrogram features from audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio time series.
            sr (int): Sample rate of the audio.
            
        Returns:
            numpy.ndarray: Extracted spectrogram features.
        """
        try:
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sr, 
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Reshape for CNN input: (n_mels, time, 1)
            spec_features = log_mel_spec.reshape(self.n_mels, -1, 1)
            
            return spec_features
        
        except Exception as e:
            logger.error(f"Error extracting spectrogram: {e}")
            raise
    
    def process_audio_file(self, file_path, feature_type='both'):
        """
        Process a single audio file to extract features.
        
        Args:
            file_path (str): Path to the audio file.
            feature_type (str): Type of features to extract ('mfcc', 'spectrogram', or 'both').
            
        Returns:
            dict: Dictionary containing the extracted features.
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(file_path, sr=None)
            
            features = {}
            
            if feature_type in ['mfcc', 'both']:
                features['mfcc'] = self.extract_mfcc(audio_data, sr)
                
            if feature_type in ['spectrogram', 'both']:
                features['spectrogram'] = self.extract_spectrogram(audio_data, sr)
            
            return features
        
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            raise
    
    def process_dataset(self, dataset, feature_type='both', max_length=None):
        """
        Process a dataset to extract features from all audio files.
        
        Args:
            dataset (pandas.DataFrame): Dataset containing audio paths and labels.
            feature_type (str): Type of features to extract ('mfcc', 'spectrogram', or 'both').
            max_length (int): Maximum length for spectrograms. If None, will use the longest found.
            
        Returns:
            tuple: Tuple containing features and labels.
        """
        mfcc_features = []
        spec_features = []
        labels = []
        spec_lengths = []
        
        try:
            # Identify the label column
            label_column = None
            for potential_col in ['labels', 'label', 'emotion', 'emotion_id']:
                if potential_col in dataset.columns:
                    label_column = potential_col
                    break
            
            if label_column is None:
                logger.warning("No label column found in dataset. Using the first column as a fallback.")
                label_column = dataset.columns[0]
            
            logger.info(f"Using '{label_column}' as the label column for feature extraction")
            
            # First pass to extract features and determine max spectrogram length if needed
            for idx, row in dataset.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing audio {idx+1}/{len(dataset)}")
                
                # Handle different audio path formats
                if isinstance(row['audio'], dict) and 'path' in row['audio']:
                    audio_path = row['audio']['path']
                elif isinstance(row['audio'], str):
                    audio_path = row['audio']
                else:
                    logger.warning(f"Unexpected audio format for row {idx}, skipping")
                    continue
                
                # Get label from the identified column
                label = row[label_column]
                
                features = self.process_audio_file(audio_path, feature_type)
                
                if 'mfcc' in features:
                    mfcc_features.append(features['mfcc'])
                
                if 'spectrogram' in features:
                    spec = features['spectrogram']
                    spec_features.append(spec)
                    spec_lengths.append(spec.shape[1])  # Store the length for padding later
                
                labels.append(label)
            
            # Convert lists to numpy arrays
            labels = np.array(labels)
            
            result = {}
            
            if mfcc_features:
                mfcc_features = np.array(mfcc_features)
                result['mfcc'] = mfcc_features
                
            if spec_features:
                # Determine the max length to use for padding
                if max_length is None:
                    max_length = max(spec_lengths)
                
                logger.info(f"Padding spectrograms to length {max_length}")
                
                # Pad spectrograms to the same length
                padded_specs = []
                
                for spec in spec_features:
                    if spec.shape[1] < max_length:
                        padded_spec = np.pad(
                            spec, 
                            ((0, 0), (0, max_length - spec.shape[1]), (0, 0)),
                            mode='constant'
                        )
                    elif spec.shape[1] > max_length:
                        # If the spectrogram is longer than max_length, truncate it
                        padded_spec = spec[:, :max_length, :]
                    else:
                        padded_spec = spec
                    
                    padded_specs.append(padded_spec)
                
                spec_features = np.array(padded_specs)
                result['spectrogram'] = spec_features
                result['max_length'] = max_length  # Store the max_length for future use
            
            result['labels'] = labels
            
            logger.info(f"Processed {len(dataset)} audio files successfully")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise
    
    def normalize_features(self, features, feature_type='both', fit=True):
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            features (dict): Dictionary containing features and labels.
            feature_type (str): Type of features to normalize ('mfcc', 'spectrogram', or 'both').
            fit (bool): Whether to fit the scaler on the data or just transform.
            
        Returns:
            dict: Dictionary containing normalized features and labels.
        """
        normalized_features = {}
        
        try:
            if 'mfcc' in features and feature_type in ['mfcc', 'both']:
                mfcc_data = features['mfcc']
                
                # Reshape for normalization
                original_shape = mfcc_data.shape
                mfcc_data_2d = mfcc_data.reshape(-1, original_shape[-1])
                
                # Normalize
                if fit:
                    normalized_mfcc = self.scaler_mfcc.fit_transform(mfcc_data_2d)
                else:
                    normalized_mfcc = self.scaler_mfcc.transform(mfcc_data_2d)
                
                # Reshape back
                normalized_mfcc = normalized_mfcc.reshape(original_shape)
                
                normalized_features['mfcc'] = normalized_mfcc
            
            if 'spectrogram' in features and feature_type in ['spectrogram', 'both']:
                spec_data = features['spectrogram']
                
                # Reshape for normalization
                original_shape = spec_data.shape
                spec_data_2d = spec_data.reshape(-1, 1)
                
                # Normalize
                if fit:
                    normalized_spec = self.scaler_spec.fit_transform(spec_data_2d)
                else:
                    normalized_spec = self.scaler_spec.transform(spec_data_2d)
                
                # Reshape back
                normalized_spec = normalized_spec.reshape(original_shape)
                
                normalized_features['spectrogram'] = normalized_spec
            
            normalized_features['labels'] = features['labels']
            
            logger.info(f"Features normalized successfully")
            
            return normalized_features
        
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load dataset
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    train_data, val_data, test_data = data_loader.split_dataset()
    
    # Extract features from a small subset for demonstration
    feature_extractor = FeatureExtractor()
    
    # Process a small subset of the training data
    small_subset = train_data.head(5)
    features = feature_extractor.process_dataset(small_subset)
    
    # Normalize features
    normalized_features = feature_extractor.normalize_features(features)
    
    # Print shapes
    if 'mfcc' in normalized_features:
        print(f"MFCC features shape: {normalized_features['mfcc'].shape}")
    
    if 'spectrogram' in normalized_features:
        print(f"Spectrogram features shape: {normalized_features['spectrogram'].shape}")
    
    print(f"Labels shape: {normalized_features['labels'].shape}")