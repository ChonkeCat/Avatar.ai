import os
from PIL import Image

# --- SETTINGS ---
input_root = "C:/Projects/CNN/data"           # original dataset
output_root = "C:/Projects/CNN/data_80x80"    # resized dataset

target_size = (80, 80)  # (width, height)

# --- CREATE OUTPUT ROOT ---
os.makedirs(output_root, exist_ok=True)

# --- PROCESS ---
for subfolder in os.listdir(input_root):
    subfolder_path = os.path.join(input_root, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Create matching output subfolder
    out_subfolder = os.path.join(output_root, subfolder)
    os.makedirs(out_subfolder, exist_ok=True)

    # Process images
    for filename in os.listdir(subfolder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(subfolder_path, filename)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # ensures consistent channels

                # Resize using downscale-friendly resample
                img_resized = img.resize(target_size, resample=Image.Resampling.BOX)

                # Save resized image
                out_path = os.path.join(out_subfolder, filename)
                img_resized.save(out_path)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

print("All images resized to 80x80!")
