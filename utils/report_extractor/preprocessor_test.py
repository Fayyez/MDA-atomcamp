import logging
from PIL import Image
from .preprocessing.image_preprocessor import ImagePreprocessor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Create directories if they don't exist
input_dir = Path("input_reports")
output_dir = Path("output_reports")
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Initialize preprocessor
preprocessor = ImagePreprocessor()

# Process all images in input folder
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
image_files = [f for f in input_dir.iterdir() if f.suffix.lower()
               in image_extensions]

if not image_files:
    print(f"No images found in {input_dir}/")
else:
    for image_path in image_files:
        try:
            print(f"\nProcessing {image_path.name}...")
            img = Image.open(image_path).convert("RGB")

            # Smart preprocess: single pass quality assessment + preprocessing
            clean_img, quality = preprocessor.smart_preprocess(img)
            print(
                f"  Sharpness: {quality.sharpness:.1f} | Contrast: {quality.contrast:.1f} | Brightness: {quality.brightness:.1f}")
            if quality.is_blurry:
                print("  ⚠ Image is blurry — sharpening will be applied")
            if quality.is_low_contrast:
                print("  ⚠ Low contrast detected — enhancement will be applied")

            output_path = output_dir / f"clean_{image_path.name}"
            clean_img.save(output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")

    print(f"\nProcessing complete! Results saved to {output_dir}/")
