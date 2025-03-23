import os
import glob
import cv2
import numpy as np
import torch
import lpips
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as calculate_ssim
import matplotlib.pyplot as plt
from scipy import stats
import time
from skimage.metrics import mean_squared_error
import sys

# Force UTF-8 encoding for output if possible
if sys.stdout.encoding != 'utf-8':
    try:
        # Try to set UTF-8 encoding for Windows
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        # If that fails, we'll make sure to use only ASCII in our output
        pass

# Import pyiqa for NIQE calculation
import pyiqa

# Load LPIPS model
lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# Load NIQE model from pyiqa
niqe_model = pyiqa.create_metric('niqe', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define paths
high_dir = './datasets/eval15/high'
low_dir = './datasets/eval15/low'
enlighten_dir = './datasets/eval15/Enlighten'  # Directory with pre-processed EnlightenGAN images
pretrained_dir = './datasets/eval15/pretrained' # Directory with pretrained model results
results_dir = './datasets/eval15/results'

os.makedirs(results_dir, exist_ok=True)

# Calculate NIQE score using pyiqa
def calculate_quality(img):
    """
    Calculate NIQE quality score.
    NIQE is a no-reference image quality metric that doesn't require a reference image.
    """
    try:
        # Save image to a temporary file since pyiqa works well with file paths
        temp_file = 'temp_for_niqe.jpg'
        cv2.imwrite(temp_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Use pyiqa with just the test image
        with torch.no_grad():
            niqe_score = niqe_model(temp_file)
            if isinstance(niqe_score, torch.Tensor):
                niqe_score = niqe_score.item()
        
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return niqe_score
    except Exception as e:
        print(f"Error in calculate_quality: {str(e)}")
        return float('nan')  # Return NaN on error

# Define CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Gamma correction function
def apply_gamma_correction(img, gamma=2.0):
    # Normalize to [0,1]
    img_norm = img.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(img_norm, 1.0/gamma)
    
    # Convert back to [0,255] and uint8
    corrected = (corrected * 255.0).astype(np.uint8)
    
    return corrected

# Functions to calculate metrics
def calculate_psnr(img1, img2):
    return psnr(img1, img2)

def calculate_lpips(img1, img2):
    tensor1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1.0
    tensor2 = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1.0
    
    if torch.cuda.is_available():
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()
    
    with torch.no_grad():
        lpips_score = lpips_model(tensor1, tensor2).item()
    
    return lpips_score

def apply_clahe(img, apply_gamma=True, gamma=1.5):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    l_clahe = clahe.apply(l)
    
    # Merge the channels back
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Apply gamma correction if requested
    if apply_gamma:
        enhanced = apply_gamma_correction(enhanced, gamma)
    
    return enhanced

# Get list of image files
high_imgs = sorted(glob.glob(os.path.join(high_dir, '*.jpg')) + 
                   glob.glob(os.path.join(high_dir, '*.png')) +
                   glob.glob(os.path.join(high_dir, '*.bmp')))

low_imgs = sorted(glob.glob(os.path.join(low_dir, '*.jpg')) + 
                  glob.glob(os.path.join(low_dir, '*.png')) +
                  glob.glob(os.path.join(low_dir, '*.bmp')))

enlighten_imgs = sorted(glob.glob(os.path.join(enlighten_dir, '*.jpg')) + 
                        glob.glob(os.path.join(enlighten_dir, '*.png')) +
                        glob.glob(os.path.join(enlighten_dir, '*.bmp')))

pretrained_imgs = sorted(glob.glob(os.path.join(pretrained_dir, '*.jpg')) + 
                        glob.glob(os.path.join(pretrained_dir, '*.png')) +
                        glob.glob(os.path.join(pretrained_dir, '*.bmp')))

print(f"Found {len(high_imgs)} high-quality images and {len(low_imgs)} low-quality images")
print(f"Found {len(enlighten_imgs)} EnlightenGAN processed images")
print(f"Found {len(pretrained_imgs)} pretrained model processed images")

# Ensure matching filenames across all directories
high_filenames = [os.path.basename(img) for img in high_imgs]
low_filenames = [os.path.basename(img) for img in low_imgs]
enlighten_filenames = [os.path.basename(img) for img in enlighten_imgs]
pretrained_filenames = [os.path.basename(img) for img in pretrained_imgs]

# Find common filenames
common_files = list(set(high_filenames).intersection(set(low_filenames)).intersection(set(enlighten_filenames)).intersection(set(pretrained_filenames)))
print(f"Found {len(common_files)} common files across all directories to process")

if len(common_files) == 0:
    print("No matching files found. Checking if EnlightenGAN images have different naming convention...")
    # Try to match by removing any prefixes/suffixes in enlighten_filenames
    base_low_names = [os.path.splitext(name)[0] for name in low_filenames]
    
    # Create a mapping from base low names to full filenames
    enlighten_mapping = {}
    for enlighten_file in enlighten_imgs:
        for base_name in base_low_names:
            if base_name in os.path.basename(enlighten_file):
                enlighten_mapping[base_name] = enlighten_file
                break
    
    if enlighten_mapping:
        print(f"Found {len(enlighten_mapping)} matches using partial name matching")
        # Filter images based on the mapping
        processed_files = []
        processed_high_imgs = []
        processed_low_imgs = []
        processed_enlighten_imgs = []
        
        for i, (high_path, low_path) in enumerate(zip(high_imgs, low_imgs)):
            base_name = os.path.splitext(os.path.basename(low_path))[0]
            if base_name in enlighten_mapping:
                processed_files.append(base_name)
                processed_high_imgs.append(high_path)
                processed_low_imgs.append(low_path)
                processed_enlighten_imgs.append(enlighten_mapping[base_name])
        
        high_imgs = processed_high_imgs
        low_imgs = processed_low_imgs
        enlighten_imgs = processed_enlighten_imgs
    else:
        print("Error: Cannot match EnlightenGAN images with original images.")
        import sys
        sys.exit(1)
else:
    # Filter images to only include common files
    high_imgs = [img for img in high_imgs if os.path.basename(img) in common_files]
    low_imgs = [img for img in low_imgs if os.path.basename(img) in common_files]
    enlighten_imgs = [img for img in enlighten_imgs if os.path.basename(img) in common_files]
    pretrained_imgs = [img for img in pretrained_imgs if os.path.basename(img) in common_files]

# Create empty lists to store metrics
enlighten_psnr = []
enlighten_ssim = []
enlighten_lpips = []
enlighten_niqe = []

clahe_psnr = []
clahe_ssim = []
clahe_lpips = []
clahe_niqe = []

pretrained_psnr = []
pretrained_ssim = []
pretrained_lpips = []
pretrained_niqe = []

original_niqe = []
reference_niqe = []  # Add a list to store reference image NIQE scores

# Debug print for verification
print("\nProcessing the following image files:")
for i, (high_path, low_path, enlighten_path, pretrained_path) in enumerate(zip(high_imgs, low_imgs, enlighten_imgs, pretrained_imgs)):
    print(f"{i+1}. High: {os.path.basename(high_path)}, Low: {os.path.basename(low_path)}, EnlightenGAN: {os.path.basename(enlighten_path)}, Pretrained: {os.path.basename(pretrained_path)}")

# Process each image triplet
for i, (high_path, low_path, enlighten_path, pretrained_path) in enumerate(zip(high_imgs, low_imgs, enlighten_imgs, pretrained_imgs)):
    print(f"\nProcessing image {i+1}/{len(high_imgs)}: {os.path.basename(low_path)}")
    
    # Read images
    high_img = cv2.imread(high_path)
    low_img = cv2.imread(low_path)
    enlighten_img = cv2.imread(enlighten_path)
    pretrained_img = cv2.imread(pretrained_path)
    
    if high_img is None:
        print(f"ERROR: Could not read high image: {high_path}")
        continue
    if low_img is None:
        print(f"ERROR: Could not read low image: {low_path}")
        continue
    if enlighten_img is None:
        print(f"ERROR: Could not read enlighten image: {enlighten_path}")
        continue
    if pretrained_img is None:
        print(f"ERROR: Could not read pretrained image: {pretrained_path}")
        continue
    
    # Ensure images have the same size
    print(f"Image shapes - High: {high_img.shape}, Low: {low_img.shape}, Enlighten: {enlighten_img.shape}, Pretrained: {pretrained_img.shape}")
    
    if high_img.shape != low_img.shape:
        print(f"Resizing high image to match low image dimensions")
        high_img = cv2.resize(high_img, (low_img.shape[1], low_img.shape[0]))
    
    if enlighten_img.shape != low_img.shape:
        print(f"Resizing enlighten image to match low image dimensions")
        enlighten_img = cv2.resize(enlighten_img, (low_img.shape[1], low_img.shape[0]))
    
    if pretrained_img.shape != low_img.shape:
        print(f"Resizing pretrained image to match low image dimensions")
        pretrained_img = cv2.resize(pretrained_img, (low_img.shape[1], low_img.shape[0]))
    
    # Calculate NIQE for reference (high-quality) image
    try:
        ref_niqe = calculate_quality(high_img)
        reference_niqe.append(ref_niqe)
        print(f"  Reference    - NIQE: {ref_niqe:.4f}")
    except Exception as e:
        print(f"Error calculating reference quality metrics: {str(e)}")
        reference_niqe.append(float('nan'))
    
    # Calculate NIQE for original low-light image
    try:
        low_niqe = calculate_quality(low_img)
        original_niqe.append(low_niqe)
        print(f"  Original low-light     - NIQE: {low_niqe:.4f}")
    except Exception as e:
        print(f"Error calculating original quality metrics: {str(e)}")
        original_niqe.append(float('nan'))
    
    # Apply CLAHE
    start_time = time.time()
    gamma_value = 2.0  # Define gamma value here for easy adjustment
    clahe_result = apply_clahe(low_img, apply_gamma=True, gamma=gamma_value)
    clahe_time = time.time() - start_time
    print(f"CLAHE + Gamma (g={gamma_value}) processing time: {clahe_time:.4f} seconds")
    
    # Calculate metrics for EnlightenGAN (using pre-processed images)
    try:
        current_psnr = calculate_psnr(high_img, enlighten_img)
        current_ssim = calculate_ssim(high_img, enlighten_img, channel_axis=2)
        current_lpips = calculate_lpips(high_img, enlighten_img)
        current_niqe = calculate_quality(enlighten_img)
        
        enlighten_psnr.append(current_psnr)
        enlighten_ssim.append(current_ssim)
        enlighten_lpips.append(current_lpips)
        enlighten_niqe.append(current_niqe)
        
        print(f"  EnlightenGAN - PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}, NIQE: {current_niqe:.4f}")
    except Exception as e:
        print(f"Error calculating EnlightenGAN metrics: {str(e)}")
    
    # Calculate metrics for CLAHE
    try:
        current_psnr = calculate_psnr(high_img, clahe_result)
        current_ssim = calculate_ssim(high_img, clahe_result, channel_axis=2)
        current_lpips = calculate_lpips(high_img, clahe_result)
        current_niqe = calculate_quality(clahe_result)
        
        clahe_psnr.append(current_psnr)
        clahe_ssim.append(current_ssim)
        clahe_lpips.append(current_lpips)
        clahe_niqe.append(current_niqe)
        
        print(f"  CLAHE+Gamma  - PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}, NIQE: {current_niqe:.4f}")
    except Exception as e:
        print(f"Error calculating CLAHE metrics: {str(e)}")
    
    # Calculate metrics for pretrained model
    try:
        current_psnr = calculate_psnr(high_img, pretrained_img)
        current_ssim = calculate_ssim(high_img, pretrained_img, channel_axis=2)
        current_lpips = calculate_lpips(high_img, pretrained_img)
        current_niqe = calculate_quality(pretrained_img)
        
        pretrained_psnr.append(current_psnr)
        pretrained_ssim.append(current_ssim)
        pretrained_lpips.append(current_lpips)
        pretrained_niqe.append(current_niqe)
        
        print(f"  Pretrained - PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}, NIQE: {current_niqe:.4f}")
    except Exception as e:
        print(f"Error calculating pretrained metrics: {str(e)}")
    
    # Create concatenated image (original low | original high | EnlightenGAN | CLAHE | Pretrained)
    concat_img = np.hstack((low_img, high_img, enlighten_img, clahe_result, pretrained_img))
    
    # Add labels to the image
    img_with_labels = concat_img.copy()
    
    # Get text size for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    small_font_scale = 0.7  # For the metrics text
    
    # Define the text color (BGR format)
    text_color = (0, 0, 255)  # REd color in BGR
    
    # Low-light Image
    text = "Low-light"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (low_img.shape[1]//2 - text_size[0]//2, 30)
    cv2.putText(img_with_labels, text, position, font, font_scale, text_color, thickness)
    # Add NIQE value
    metrics_text = f"NIQE: {low_niqe:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1]//2 - text_size[0]//2, 60)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    
    # High Image
    text = "Reference"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1]//2 - text_size[0]//2, 30)
    cv2.putText(img_with_labels, text, position, font, font_scale, text_color, thickness)
    # Add NIQE value
    metrics_text = f"NIQE: {ref_niqe:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1]//2 - text_size[0]//2, 60)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    
    # Enlighten Image
    text = "EnlightenGAN"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1]//2 - text_size[0]//2, 30)
    cv2.putText(img_with_labels, text, position, font, font_scale, text_color, thickness)
    # Add metrics
    metrics_text = f"PSNR: {enlighten_psnr[-1]:.2f}, SSIM: {enlighten_ssim[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1]//2 - text_size[0]//2, 60)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    metrics_text = f"LPIPS: {enlighten_lpips[-1]:.2f}, NIQE: {enlighten_niqe[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1]//2 - text_size[0]//2, 80)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    
    # CLAHE Image
    text = f"CLAHE+Gamma (g={gamma_value})"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1]//2 - text_size[0]//2, 30)
    cv2.putText(img_with_labels, text, position, font, font_scale, text_color, thickness)
    # Add metrics
    metrics_text = f"PSNR: {clahe_psnr[-1]:.2f}, SSIM: {clahe_ssim[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1]//2 - text_size[0]//2, 60)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    metrics_text = f"LPIPS: {clahe_lpips[-1]:.2f}, NIQE: {clahe_niqe[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1]//2 - text_size[0]//2, 80)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    
    # Pretrained Image
    text = "Pretrained"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1] + pretrained_img.shape[1]//2 - text_size[0]//2, 30)
    cv2.putText(img_with_labels, text, position, font, font_scale, text_color, thickness)
    # Add metrics
    metrics_text = f"PSNR: {pretrained_psnr[-1]:.2f}, SSIM: {pretrained_ssim[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1] + pretrained_img.shape[1]//2 - text_size[0]//2, 60)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    metrics_text = f"LPIPS: {pretrained_lpips[-1]:.2f}, NIQE: {pretrained_niqe[-1]:.2f}"
    text_size = cv2.getTextSize(metrics_text, font, small_font_scale, thickness)[0]
    position = (low_img.shape[1] + high_img.shape[1] + enlighten_img.shape[1] + clahe_result.shape[1] + pretrained_img.shape[1]//2 - text_size[0]//2, 80)
    cv2.putText(img_with_labels, metrics_text, position, font, small_font_scale, text_color, thickness)
    
    # Save labeled image
    filename = os.path.basename(low_path)
    base_filename, _ = os.path.splitext(filename)
    save_path = os.path.join(results_dir, f"compare_{base_filename}.jpg")
    cv2.imwrite(save_path, img_with_labels, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Saved comparison image to {save_path}")

# Check if we have valid metrics
if len(enlighten_psnr) == 0 or len(clahe_psnr) == 0 or len(pretrained_psnr) == 0:
    print("\nError: No valid metrics were calculated. Check the image files and paths.")
    import sys
    sys.exit(1)

# Calculate average metrics
avg_enlighten_psnr = np.mean(enlighten_psnr)
avg_enlighten_ssim = np.mean(enlighten_ssim)
avg_enlighten_lpips = np.mean(enlighten_lpips)
avg_enlighten_niqe = np.mean(enlighten_niqe)

avg_clahe_psnr = np.mean(clahe_psnr)
avg_clahe_ssim = np.mean(clahe_ssim)
avg_clahe_lpips = np.mean(clahe_lpips)
avg_clahe_niqe = np.mean(clahe_niqe)

avg_pretrained_psnr = np.mean(pretrained_psnr)
avg_pretrained_ssim = np.mean(pretrained_ssim)
avg_pretrained_lpips = np.mean(pretrained_lpips)
avg_pretrained_niqe = np.mean(pretrained_niqe)

avg_original_niqe = np.mean(original_niqe)
avg_reference_niqe = np.mean(reference_niqe)  # Calculate average reference NIQE

# Print results table
print("\n" + "="*80)
print("{:<15} {:<12} {:<12} {:<12} {:<12}".format("Method", "PSNR ^", "SSIM ^", "LPIPS v", "NIQE v"))
print("-"*80)
print("{:<15} {:<12} {:<12} {:<12} {:<12.4f}".format("Reference", "N/A", "N/A", "N/A", avg_reference_niqe))
print("{:<15} {:<12} {:<12} {:<12} {:<12.4f}".format("Original low-light", "N/A", "N/A", "N/A", avg_original_niqe))
print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format("EnlightenGAN", avg_enlighten_psnr, avg_enlighten_ssim, avg_enlighten_lpips, avg_enlighten_niqe))
print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(f"CLAHE+g({gamma_value})", avg_clahe_psnr, avg_clahe_ssim, avg_clahe_lpips, avg_clahe_niqe))
print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format("Pretrained", avg_pretrained_psnr, avg_pretrained_ssim, avg_pretrained_lpips, avg_pretrained_niqe))
print("="*80)

# Save results to CSV
csv_path = os.path.join(results_dir, "evaluation_results.csv")
try:
    with open(csv_path, 'w', encoding='ascii', errors='replace') as f:
        f.write("Image,Method,PSNR,SSIM,LPIPS,NIQE\n")
        
        for i, img_path in enumerate(low_imgs):
            img_name = os.path.basename(img_path)
            if i < len(enlighten_psnr):  # Make sure we have metrics for this image
                f.write(f"{img_name},Reference,N/A,N/A,N/A,{reference_niqe[i]:.6f}\n")
                f.write(f"{img_name},Original low-light,N/A,N/A,N/A,{original_niqe[i]:.6f}\n")
                f.write(f"{img_name},EnlightenGAN,{enlighten_psnr[i]:.6f},{enlighten_ssim[i]:.6f},{enlighten_lpips[i]:.6f},{enlighten_niqe[i]:.6f}\n")
                f.write(f"{img_name},CLAHE+g({gamma_value}),{clahe_psnr[i]:.6f},{clahe_ssim[i]:.6f},{clahe_lpips[i]:.6f},{clahe_niqe[i]:.6f}\n")
                f.write(f"{img_name},Pretrained,{pretrained_psnr[i]:.6f},{pretrained_ssim[i]:.6f},{pretrained_lpips[i]:.6f},{pretrained_niqe[i]:.6f}\n")
    
    print(f"\nResults saved to {csv_path}")
    print(f"Processed images saved to {results_dir}")
except Exception as e:
    print(f"Error saving CSV: {str(e)}")

# Create plots
try:
    # Configure matplotlib to use safe fonts
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    plt.figure(figsize=(15, 10))
    
    clahe_label = f"CLAHE+g({gamma_value})"
    
    # PSNR comparison
    plt.subplot(2, 2, 1)
    plt.bar(['EnlightenGAN', clahe_label, 'Pretrained'], [avg_enlighten_psnr, avg_clahe_psnr, avg_pretrained_psnr])
    plt.ylabel('PSNR (higher is better)')
    plt.title('PSNR Comparison')
    
    # SSIM comparison
    plt.subplot(2, 2, 2)
    plt.bar(['EnlightenGAN', clahe_label, 'Pretrained'], [avg_enlighten_ssim, avg_clahe_ssim, avg_pretrained_ssim])
    plt.ylabel('SSIM (higher is better)')
    plt.title('SSIM Comparison')
    
    # LPIPS comparison
    plt.subplot(2, 2, 3)
    plt.bar(['EnlightenGAN', clahe_label, 'Pretrained'], [avg_enlighten_lpips, avg_clahe_lpips, avg_pretrained_lpips])
    plt.ylabel('LPIPS (lower is better)')
    plt.title('LPIPS Comparison')
    
    # NIQE comparison
    plt.subplot(2, 2, 4)
    plt.bar(['Reference', 'Original low-light', 'EnlightenGAN', clahe_label, 'Pretrained'], 
            [avg_reference_niqe, avg_original_niqe, avg_enlighten_niqe, avg_clahe_niqe, avg_pretrained_niqe])
    plt.ylabel('NIQE (lower is better)')
    plt.title('NIQE Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "metrics_comparison.jpg"), format='jpeg', dpi=300)
    print(f"Comparison plots saved to {os.path.join(results_dir, 'metrics_comparison.jpg')}")
except Exception as e:
    print(f"Error creating plots: {str(e)}") 