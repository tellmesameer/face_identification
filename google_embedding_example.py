"""
Image Embedding Generator Module

This module provides functionality to generate embeddings for images using Google's Gemini API.
It includes support for various image formats, retry logic with exponential backoff,
and timing measurements for performance monitoring.
"""

import os
import time
import random
from typing import Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types


def _get_mime_type(file_path: str) -> str:
    """
    Get the MIME type based on file extension.

    Args:
        file_path: Path to the image file

    Returns:
        MIME type string for the image format
    """
    _, extension = os.path.splitext(file_path)
    extension_lower = extension.lower()

    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }

    return mime_type_map.get(extension_lower, 'image/jpeg')


def generate_image_embedding_with_retry(
    image_path: str,
    model_name: str = 'gemini-embedding-2-preview',
    max_retries: int = 5,
    base_delay: float = 1.0
) -> Tuple[list, float]:
    """
    Generate embeddings for an image with retry logic and timing information.

    This function reads an image file, determines its MIME type, and generates
    embeddings using the specified Gemini model. It implements exponential
    backoff retry logic for handling transient API errors.

    Args:
        image_path: Path to the image file to process
        model_name: Name of the embedding model to use (default: 'gemini-embedding-2-preview')
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds before first retry (default: 1.0)

    Returns:
        A tuple containing:
        - embeddings: List of embedding vectors
        - response_time: Processing time in seconds

    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the API key is not configured
        Exception: If all retry attempts fail
    """
    # Load environment variables to get the API key
    load_dotenv()

    # Validate that API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    # Validate image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read the image file content
    with open(image_path, 'rb') as file_handle:
        image_data = file_handle.read()

    # Determine the appropriate MIME type for the image
    mime_type = _get_mime_type(image_path)

    # Store the last encountered exception for re-raising after retries
    last_encountered_exception = None

    # Attempt to generate embeddings with retry logic
    for attempt_number in range(max_retries + 1):  # Include initial attempt
        try:
            # Record start time for performance measurement
            start_timestamp = time.time()

            # Call the Gemini API to generate embeddings
            result = client.models.embed_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type,
                    ),
                ]
            )

            # Calculate total processing time
            end_timestamp = time.time()
            processing_duration = end_timestamp - start_timestamp

            print(f"Response Time: {processing_duration:.3f} seconds")
            return result.embeddings, processing_duration

        except Exception as error:
            # Store the exception for potential re-raising later
            last_encountered_exception = error

            # Check if we should retry or give up
            if attempt_number < max_retries:
                # Calculate delay with exponential backoff and random jitter
                calculated_delay = base_delay * (2 ** attempt_number) + random.uniform(0, 1)
                print(f"Attempt {attempt_number + 1} failed: {str(error)}")
                print(f"Retrying in {calculated_delay:.2f} seconds...")
                time.sleep(calculated_delay)
            else:
                print(f"All {max_retries + 1} attempts failed.")
                raise error

    # This line should theoretically never be reached due to the raise above,
    # but included for completeness
    if last_encountered_exception:
        raise last_encountered_exception


# Example usage:
if __name__ == "__main__":
    embeddings, response_time = generate_image_embedding_with_retry('example.jpg')

    # Print number of embeddings
    print(f"Number of embeddings: {len(embeddings)}")

    # Print dimension of first embedding vector
    if embeddings:
        vector = embeddings[0].values  # Gemini embedding vector
        print(f"Embedding dimension: {len(vector)}")

    print(f"Generated embeddings: {embeddings}")


# 
# Embedding dimension: 3072 defines the fixed vector size produced by gemini-embedding-2-preview.

# Implications:

# Each image → 3072-length float vector
# Storage shape → (n_images, 3072)
# Similarity ops → cosine / dot product directly valid
# Indexing systems (FAISS, Annoy, etc.) must use dim=3072