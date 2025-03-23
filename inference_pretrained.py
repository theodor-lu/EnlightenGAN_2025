import os
import cv2
import numpy as np
from enlighten_inference import EnlightenOnnxModel
import multiprocessing

def main():
    # Define input and output directories
    input_dir = 'datasets/test_A'
    output_dir = 'predict_pretrained'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model
    # By default, CUDAExecutionProvider is used if available
    try:
        model = EnlightenOnnxModel()
    except Exception as e:
        print(f"Failed to initialize with CUDA, falling back to CPU: {e}")
        model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    
    # Get all image files recursively
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process all images
    for input_path in image_files:
        print(f"Processing {input_path}...")
        
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read {input_path}")
                continue
                
            # Process the image
            processed = model.predict(img)
            
            # Save to output directory with the filename only (not the full path)
            filename = os.path.basename(input_path)
            
            # If there are duplicate filenames, add a suffix
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, filename)
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                counter += 1
                
            cv2.imwrite(output_path, processed)
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    print("Processing complete!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 