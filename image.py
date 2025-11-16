import os
from PIL import Image

# --- SETTINGS ---
input_root = "C:/Projects/CNN/data"    # root folder with subfolders of images
output_root = "C:/Projects/CNN/data_resized"  # where resized images will go

target_size = (160, 120)  # (width, height) -> half of 320x240

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
        if not filename.lower().endswith((".jpg", ".jpeg")):
            continue
        img_path = os.path.join(subfolder_path, filename)
        img = Image.open(img_path)
        
        # Resize using averaging (ANTIALIAS)
        img_resized = img.resize(target_size, resample=Image.Resampling.BOX)
        
        # Save resized image
        out_path = os.path.join(out_subfolder, filename)
        img_resized.save(out_path)

print("All images processed and resized!")
