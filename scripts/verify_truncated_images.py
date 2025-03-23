
import os
from PIL import Image

def is_image_file(filename):
    # List of valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    return filename.lower().endswith(valid_extensions)

def main():
    datasets_dir = "datasets"
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                try:
                    # Attempt to open and verify image integrity
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    # If an exception is raised then the image is likely truncated or corrupted
                    print(f"Deleting truncated/corrupted image: {file_path}\nReason: {e}")
                    os.remove(file_path)

if __name__ == "__main__":
    main() 