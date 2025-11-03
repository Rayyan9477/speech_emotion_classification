#!/usr/bin/env python3
"""
Model Splitter - Splits large model files into smaller pieces for GitHub storage
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_file(file_path, chunk_size_mb=50):
    """
    Split a file into chunks of specified size in MB

    Args:
        file_path (str): Path to the file to split
        chunk_size_mb (int): Size of each chunk in MB

    Returns:
        list: List of paths to the created chunks
    """
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    base_name = os.path.splitext(file_path)[0]
    extension = os.path.splitext(file_path)[1]

    chunks = []
    chunk_num = 1

    with open(file_path, 'rb') as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break

            chunk_path = f"{base_name}_part{chunk_num}{extension}"
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)

            chunks.append(chunk_path)
            logger.info(f"Created chunk {chunk_num}: {chunk_path} ({len(chunk_data)} bytes)")
            chunk_num += 1

    return chunks

def combine_files(chunk_paths, output_path):
    """
    Combine multiple file chunks back into a single file

    Args:
        chunk_paths (list): List of paths to the chunks in order
        output_path (str): Path for the combined output file

    Returns:
        str: Path to the combined file
    """
    with open(output_path, 'wb') as output_file:
        for chunk_path in chunk_paths:
            with open(chunk_path, 'rb') as chunk_file:
                output_file.write(chunk_file.read())
            logger.info(f"Combined chunk: {chunk_path}")

    return output_path

def split_model_files(models_dir="models", chunk_size_mb=50):
    """
    Split all large model files in the models directory

    Args:
        models_dir (str): Directory containing model files
        chunk_size_mb (int): Size of each chunk in MB
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.error(f"Models directory {models_dir} does not exist")
        return

    # Find all model files that are too large
    large_files = []
    for file_path in models_dir.glob("*"):
        if file_path.suffix in ['.keras', '.h5'] and file_path.stat().st_size > (chunk_size_mb * 1024 * 1024):
            large_files.append(file_path)

    if not large_files:
        logger.info("No large model files found to split")
        return

    logger.info(f"Found {len(large_files)} large model files to split")

    for file_path in large_files:
        logger.info(f"Splitting {file_path}...")
        chunks = split_file(str(file_path), chunk_size_mb)

        # Remove the original large file
        os.remove(file_path)
        logger.info(f"Removed original file: {file_path}")

        # Create a manifest file listing the chunks
        manifest_path = f"{file_path.stem}_manifest.txt"
        with open(manifest_path, 'w') as f:
            f.write(f"Original file: {file_path.name}\n")
            f.write(f"Split into {len(chunks)} parts:\n")
            for chunk in chunks:
                f.write(f"- {Path(chunk).name}\n")

        logger.info(f"Created manifest: {manifest_path}")

def get_model_chunks(base_name, models_dir="models"):
    """
    Get all chunks for a model file

    Args:
        base_name (str): Base name of the model (without extension)
        models_dir (str): Directory containing model files

    Returns:
        list: List of chunk file paths in order
    """
    models_dir = Path(models_dir)
    chunks = []

    # Look for chunk files (both .keras and .h5 extensions)
    for ext in ['.keras', '.h5']:
        for file_path in models_dir.glob(f"{base_name}_part*{ext}"):
            chunks.append(file_path)

    # Sort chunks by part number
    chunks.sort(key=lambda x: int(x.stem.split('_part')[-1]))

    return [str(chunk) for chunk in chunks]

def combine_model_file(base_name, models_dir="models"):
    """
    Combine a split model file back together

    Args:
        base_name (str): Base name of the model (without extension)
        models_dir (str): Directory containing model files

    Returns:
        str: Path to the combined model file
    """
    models_dir = Path(models_dir)

    # Find the manifest file
    manifest_path = models_dir / f"{base_name}_manifest.txt"
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return None

    # Read the manifest to get chunk info
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    # Extract original filename
    original_line = [line for line in lines if line.startswith("Original file:")][0]
    original_filename = original_line.split(": ")[1].strip()
    output_path = models_dir / original_filename

    # Get chunk filenames
    chunk_lines = [line.strip()[2:] for line in lines if line.strip().startswith("- ")]
    chunk_paths = [models_dir / chunk for chunk in chunk_lines]

    # Verify all chunks exist
    missing_chunks = [chunk for chunk in chunk_paths if not chunk.exists()]
    if missing_chunks:
        logger.error(f"Missing chunks: {missing_chunks}")
        return None

    logger.info(f"Combining {len(chunk_paths)} chunks into {output_path}")
    combined_path = combine_files([str(chunk) for chunk in chunk_paths], str(output_path))

    return combined_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split and combine model files for GitHub storage")
    parser.add_argument("--action", choices=["split", "combine"], required=True,
                       help="Action to perform: split or combine")
    parser.add_argument("--models-dir", default="models",
                       help="Directory containing model files")
    parser.add_argument("--chunk-size", type=int, default=50,
                       help="Size of each chunk in MB (default: 50)")
    parser.add_argument("--base-name", help="Base name of model file for combine action")

    args = parser.parse_args()

    if args.action == "split":
        split_model_files(args.models_dir, args.chunk_size)
    elif args.action == "combine":
        if not args.base_name:
            logger.error("--base-name is required for combine action")
            sys.exit(1)
        combine_model_file(args.base_name, args.models_dir)