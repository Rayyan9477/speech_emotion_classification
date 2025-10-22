from src.data.data_loader import DataLoader
import pandas as pd

data_loader = DataLoader()
dataset = data_loader.load_dataset()
train_data, val_data, test_data = data_loader.split_dataset()

print('Sample filenames from RAVDESS:')
for i in range(min(10, len(train_data))):
    sample = train_data.iloc[i]
    audio_path = sample['audio']['path'] if isinstance(sample['audio'], dict) else str(sample['audio'])
    print(f'{i+1}: {audio_path}')
    if 'emotion' in sample:
        print(f'   Extracted emotion: {sample["emotion"]}')
    print()

print('Emotion distribution in training set:')
if 'emotion' in train_data.columns:
    print(train_data['emotion'].value_counts())