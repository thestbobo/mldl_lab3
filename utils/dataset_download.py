import os
import urllib.request
import zipfile
import tarfile
import shutil

DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_DIR = "dataset"
DATASET_ZIP = "tiny-imagenet-200.zip"
EXTRACTED_FOLDER = "tiny-imagenet-200"


def download_dataset():
    """Download the Tiny ImageNet dataset if it does not already exist in repo."""
    if not os.path.exists(DATASET_ZIP):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
        print("Download complete.")
    else:
        print("Dataset archive already exists, skipping download.")


def extract_dataset():
    """Extract the dataset (from the .zip file) if it hasn't been extracted yet."""
    if not os.path.exists(EXTRACTED_FOLDER):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    else:
        print("Dataset already extracted, skipping extraction.")


def organize_validation_data():
    """Reorganizes the validation data into separate class folders (data preparation)."""
    val_annotations = os.path.join(EXTRACTED_FOLDER, "val", "val_annotations.txt")
    val_images_dir = os.path.join(EXTRACTED_FOLDER, "val", "images")

    with open(val_annotations, "r") as f:
        for line in f:
            parts = line.split("\t")
            filename, class_id = parts[0], parts[1]

            class_folder = os.path.join(EXTRACTED_FOLDER, "val", class_id)
            os.makedirs(class_folder, exist_ok=True)

            shutil.move(os.path.join(val_images_dir, filename), os.path.join(class_folder, filename))

    # Remove the old 'images' folder since it's now split into class directories
    shutil.rmtree(val_images_dir)
    print("Validation data organized successfully.")


def setup_dataset():
    """Main function to download, extract, and organize dataset."""
    download_dataset()
    extract_dataset()
    organize_validation_data()

    # Move dataset into the correct folder
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATASET_DIR, EXTRACTED_FOLDER)):
        shutil.move(EXTRACTED_FOLDER, DATASET_DIR)
    print(f"Dataset is ready in {DATASET_DIR}/{EXTRACTED_FOLDER}")


if __name__ == "__main__":
    setup_dataset()
