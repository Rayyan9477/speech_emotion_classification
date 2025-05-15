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
        
        # Initialize scalers for feature normalization
        self.scaler_mfcc = StandardScaler()
        self.scaler_spec = StandardScaler()
    
    def augment_audio(self, audio_data, sr, apply_all=False):
        """
        Apply data augmentation to audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio time series
            sr (int): Sample rate
            apply_all (bool): Whether to apply all augmentations or randomly choose one
            
        Returns:
            numpy.ndarray: Augmented audio data
        """
        try:
            augmented_data = audio_data.copy()
            
            augmentations = [
                (self._time_stretch, {'rate': np.random.uniform(0.8, 1.2)}),
                (self._pitch_shift, {'steps': np.random.randint(-4, 5)}),
                (self._add_noise, {'noise_factor': np.random.uniform(0.001, 0.015)}),
            ]
            
            if apply_all:
                # Apply all augmentations in sequence
                for aug_func, params in augmentations:
                    augmented_data = aug_func(augmented_data, sr, **params)
            else:
                # Randomly choose one augmentation
                aug_func, params = np.random.choice(augmentations)
                augmented_data = aug_func(augmented_data, sr, **params)
            
            return augmented_data
            
        except Exception as e:
            logger.error(f"Error in audio augmentation: {e}")
            return audio_data
    
    def _time_stretch(self, audio_data, sr, rate=1.0):
        """Time stretching augmentation"""
        try:
            return librosa.effects.time_stretch(y=audio_data, rate=rate)
        except Exception as e:
            logger.error(f"Error in time stretching: {e}")
            return audio_data
    
    def _pitch_shift(self, audio_data, sr, steps=0):
        """Pitch shifting augmentation"""
        try:
            return librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=steps)
        except Exception as e:
            logger.error(f"Error in pitch shifting: {e}")
            return audio_data
    
    def _add_noise(self, audio_data, sr, noise_factor=0.005):
        """Add white noise augmentation"""
        try:
            noise = np.random.normal(0, 1, len(audio_data))
            return audio_data + noise_factor * noise
        except Exception as e:
            logger.error(f"Error adding noise: {e}")
            return audio_data
    
    def _time_mask(self, features, max_width=40):
        """Apply time masking to spectrogram/MFCC features"""
        try:
            time_len = features.shape[1]
            width = np.random.randint(1, max_width)
            start = np.random.randint(0, time_len - width)
            
            masked_features = features.copy()
            masked_features[:, start:start+width, :] = 0
            return masked_features
            
        except Exception as e:
            logger.error(f"Error in time masking: {e}")
            return features
    
    def _freq_mask(self, features, max_width=20):
        """Apply frequency masking to spectrogram/MFCC features"""
        try:
            freq_len = features.shape[0]
            width = np.random.randint(1, max_width)
            start = np.random.randint(0, freq_len - width)
            
            masked_features = features.copy()
            masked_features[start:start+width, :, :] = 0
            return masked_features
            
        except Exception as e:
            logger.error(f"Error in frequency masking: {e}")
            return features
    
    def extract_features(self, audio_data, sr=22050, augment=False):
        """
        Extract features from raw audio data for model prediction.
        
        Args:
            audio_data (numpy.ndarray): Raw audio data
            sr (int): Sample rate of the audio, defaults to 22050 Hz
            augment (bool): Whether to apply data augmentation
            
        Returns:
            numpy.ndarray: Processed features ready for model input
            shape: (1, 128, 165, 1) for CNN model input
        """
        try:
            # Apply data augmentation if requested
            if augment:
                audio_data = self.augment_audio(audio_data, sr)
            
            # Extract mel spectrogram with fixed shape (128, 165, 1)
            mel_spec_db = self.extract_spectrogram(audio_data, sr)
            
            # Verify the shape from spectrogram extraction
            if mel_spec_db.shape != (self.n_mels, 165, 1):
                logger.warning(f"Unexpected spectrogram shape: {mel_spec_db.shape}, expected ({self.n_mels}, 165, 1)")
                
                # Fix shape if needed
                fixed_spec = np.zeros((self.n_mels, 165, 1))
                min_mels = min(mel_spec_db.shape[0], self.n_mels)
                min_frames = min(mel_spec_db.shape[1], 165)
                min_channels = min(mel_spec_db.shape[2], 1)
                fixed_spec[:min_mels, :min_frames, :min_channels] = \
                    mel_spec_db[:min_mels, :min_frames, :min_channels]
                mel_spec_db = fixed_spec
            
            # Apply frequency and time masking augmentations if requested
            if augment:
                if np.random.random() > 0.5:
                    mel_spec_db = self._freq_mask(mel_spec_db)
                if np.random.random() > 0.5:
                    mel_spec_db = self._time_mask(mel_spec_db)
            
            # Reshape to (batch_size, height, width, channels) = (1, 128, 165, 1)
            features = mel_spec_db.reshape(1, self.n_mels, 165, 1)
            
            # Verify final shape
            logger.info(f"Feature shape after processing: {features.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return np.zeros((1, self.n_mels, 165, 1))
    
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
            numpy.ndarray: Extracted spectrogram features with shape (n_mels, 165, 1).
        """
        try:
            # Ensure audio data is not empty
            if len(audio_data) == 0:
                logger.error("Empty audio data provided")
                # Return a placeholder with correct shape
                return np.zeros((self.n_mels, 165, 1))
                
            # Fix potential NaN values in audio data
            audio_data = np.nan_to_num(audio_data)
                
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sr, 
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Convert to log scale (dB) with safe handling of zeros
            # Add a small constant to prevent log(0)
            mel_spec_safe = np.maximum(mel_spec, 1e-10)
            log_mel_spec = librosa.power_to_db(mel_spec_safe, ref=np.max(mel_spec_safe))
            
            # Ensure consistent time dimension (165 frames)
            target_frames = 165
            time_frames = log_mel_spec.shape[1]
            
            if time_frames < target_frames:
                # Pad if shorter
                padding = ((0, 0), (0, target_frames - time_frames))
                log_mel_spec = np.pad(log_mel_spec, padding, mode='constant')
            elif time_frames > target_frames:
                # Truncate if longer
                log_mel_spec = log_mel_spec[:, :target_frames]
            
            # Reshape to (n_mels, time_frames, 1)
            spec_features = log_mel_spec.reshape(self.n_mels, target_frames, 1)
            
            return spec_features
        
        except Exception as e:
            logger.error(f"Error extracting spectrogram: {e}")
            # Return a placeholder with correct shape
            return np.zeros((self.n_mels, 165, 1))
    
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
                
            if feature_type in ['mel_spectrogram', 'both']:
                features['mel_spectrogram'] = self.extract_spectrogram(audio_data, sr)
            
            return features
        
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            raise
    
    def process_dataset(self, dataset, feature_type='both', max_length=None, augment=False, augment_size=1):
        """
        Process a dataset to extract features from all audio files.
        
        Args:
            dataset (pandas.DataFrame): Dataset containing audio paths and labels.
            feature_type (str): Type of features to extract ('mfcc', 'mel_spectrogram', or 'both').
            max_length (int): Maximum length for spectrograms. If None, will use the longest found.
            augment (bool): Whether to apply data augmentation
            augment_size (int): Number of augmented samples to generate per original sample
            
        Returns:
            dict: Dictionary containing extracted features and labels
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
                
                # Load and process audio file
                features = self.process_audio_file(audio_path, feature_type)
                
                # Store original features
                if 'mfcc' in features:
                    mfcc_features.append(features['mfcc'])
                if 'mel_spectrogram' in features:
                    spec = features['mel_spectrogram']
                    spec_features.append(spec)
                    spec_lengths.append(spec.shape[1])
                labels.append(label)
                
                # Generate augmented samples if requested
                if augment and augment_size > 0:
                    audio_data, sr = librosa.load(audio_path, sr=None)
                    
                    for _ in range(augment_size):
                        # Apply data augmentation
                        aug_audio = self.augment_audio(audio_data, sr)
                        
                        # Extract features from augmented audio
                        if 'mfcc' in features:
                            aug_mfcc = self.extract_mfcc(aug_audio, sr)
                            mfcc_features.append(aug_mfcc)
                        
                        if 'mel_spectrogram' in features:
                            aug_spec = self.extract_spectrogram(aug_audio, sr)
                            spec_features.append(aug_spec)
                            spec_lengths.append(aug_spec.shape[1])
                        
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
                result['mel_spectrogram'] = spec_features
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
            
            if 'mel_spectrogram' in features and feature_type in ['mel_spectrogram', 'both']:
                spec_data = features['mel_spectrogram']
                
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
                
                normalized_features['mel_spectrogram'] = normalized_spec
            
            normalized_features['labels'] = features['labels']
            
            logger.info(f"Features normalized successfully")
            
            return normalized_features
        
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise

    def save_normalization_params(self, filepath):
        """
        Save the fitted scaler parameters to a file.
        
        Args:
            filepath (str): Path to save the normalization parameters.
        """
        try:
            params = {
                'mfcc_scaler': {
                    'mean': self.scaler_mfcc.mean_,
                    'scale': self.scaler_mfcc.scale_,
                    'var': self.scaler_mfcc.var_,
                },
                'spec_scaler': {
                    'mean': self.scaler_spec.mean_,
                    'scale': self.scaler_spec.scale_,
                    'var': self.scaler_spec.var_,
                }
            }
            np.save(filepath, params)
            logger.info(f"Saved normalization parameters to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving normalization parameters: {e}")
            raise

    def load_normalization_params(self, filepath):
        """
        Load previously saved scaler parameters.
        
        Args:
            filepath (str): Path to the saved normalization parameters.
        """
        try:
            params = np.load(filepath, allow_pickle=True).item()
            
            # Restore MFCC scaler parameters
            self.scaler_mfcc.mean_ = params['mfcc_scaler']['mean']
            self.scaler_mfcc.scale_ = params['mfcc_scaler']['scale']
            self.scaler_mfcc.var_ = params['mfcc_scaler']['var']
            
            # Restore spectrogram scaler parameters
            self.scaler_spec.mean_ = params['spec_scaler']['mean']
            self.scaler_spec.scale_ = params['spec_scaler']['scale']
            self.scaler_spec.var_ = params['spec_scaler']['var']
            
            logger.info(f"Loaded normalization parameters from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading normalization parameters: {e}")
            raise

    def get_normalization_params(self):
        """
        Get the current normalization parameters.
        
        Returns:
            dict: Dictionary containing the normalization parameters for both scalers.
        """
        try:
            params = {
                'mfcc_scaler': {
                    'mean': self.scaler_mfcc.mean_.tolist(),
                    'scale': self.scaler_mfcc.scale_.tolist(),
                    'var': self.scaler_mfcc.var_.tolist(),
                },
                'spec_scaler': {
                    'mean': self.scaler_spec.mean_.tolist(),
                    'scale': self.scaler_spec.scale_.tolist(),
                    'var': self.scaler_spec.var_.tolist(),
                }
            }
            return params
            
        except Exception as e:
            logger.error(f"Error getting normalization parameters: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import librosa
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Load a test audio file (replace with your actual path)
    audio_data, sr = librosa.load("test_audio.wav", sr=None)
    
    # Extract features
    features = feature_extractor.extract_features(audio_data, sr)
    
    if features is not None:
        print(f"Extracted features shape: {features.shape}")
    else:
        print("Feature extraction failed.")