from face_recognition_system import FaceRecognitionSystem
import os
from pathlib import Path

def main():
    print("Initialize system...")
    # Initialize the entire directory tree in the current directory
    system = FaceRecognitionSystem(".")

    # Define paths
    BASE_DIR = Path("e:/face_identification")
    DATA_DIR = BASE_DIR / "data"
    REFERENCE_IMG_DIR = DATA_DIR / "reference_images"
    DATASET_DIR = "D:\\nandalal_photos"
    RESULTS_DIR = DATA_DIR / "results"
    
    # Process the entire dataset and index faces
    system.process_dataset(DATASET_DIR)

    # Search for our reference target 
    # According to face_recognition standard docs, 0.6 is a standard threshold.
    results = system.find_matches(REFERENCE_IMG_DIR, threshold=0.6)

    # Copy any matches found
    if results:
        system.copy_results(results, results_path)
    else:
        print("No results to copy.")

if __name__ == "__main__":
    main()
