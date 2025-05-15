from setuptools import setup, find_packages

setup(
    name="speech_emotion_classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'tensorflow',
        'librosa',
        'plotly',
        'numpy'
    ],
)