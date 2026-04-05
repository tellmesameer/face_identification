from face_recognition_system import FaceRecognitionSystem
from pathlib import Path

def main():
    print("Initialize system...")
    # Initialize the entire directory tree in the current directory
    system = FaceRecognitionSystem(".")

    # Define paths
    base_dir = Path("e:/face_identification")
    data_dir = base_dir / "data"
    reference_img_dir = data_dir / "reference_images"
    dataset_dir = Path(r"D:\nandalal_photos")
    results_file = data_dir / "results" / "matched_paths.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Analyze the dataset incrementally and resume from prior progress when possible.
    results = system.find_matches(reference_img_dir, dataset_dir, threshold=0.5)

    # Save matched image paths as JSON
    if results:
        system.save_result_paths(results, results_file)
    else:
        print("No matching images found.")

if __name__ == "__main__":
    main()
