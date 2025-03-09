import os
from pathlib import Path
from PIL import Image
from datasets import Dataset, load_dataset, Features, Image as DatasetImage, Value

dataset_path = "../../Desktop/dreambooth/train"
dataset_name = "graceyun/dreambooth-pixels"

def process_image(input_path, size=(512,512), icon_scale=0.5):
    try:
        img = Image.open(input_path)
        size = img.size
        
        # Handle transparency
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background.convert('RGB')
        
        # Scale down the icon
        icon_size = (int(size[0] * icon_scale), int(size[1] * icon_scale))
        img = img.resize(icon_size, Image.NEAREST)  # Use NEAREST for pixel art
        
        # Create white canvas and paste centered
        new_img = Image.new('RGB', size, (255, 255, 255))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        new_img.paste(img, (x, y))
        
        # Save as PNG
        output_path = input_path.with_suffix('.png')
        new_img.save(output_path, 'PNG')
            
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_directory(directory_path):
    image_extensions = {'.jpg', '.jpeg', '.webp', '.svg', '.png'}
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                process_image(file_path)
                
def update_metadata(directory_path):
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return
    
    csv_file = directory / "metadata.csv"
    
    if not csv_file.exists():
        print(f"Metadata file not found: {csv_file}")
        return
    
    metadata = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                file_name = parts[0]
                text = parts[1]
                metadata[file_name] = text
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == ".png":
                file_name = f"{file_path.stem}.png"
                if file_name not in metadata:
                    print(f"File not found in metadata: {file_path.stem}")
                    text = file_path.stem.replace("pixel_icon-", "")
                    text = text.replace("-", " ")
                    text = f"a PXCON, a 16-bit pixel art icon of {text}"
                    
                    csv_text = f'"{text}"' if ',' in text else text
                    with open(csv_file, 'a', encoding='utf-8') as f:
                        f.write(f"{file_name},{csv_text}\n")
  
def delete_dsstore(path):
    for file in os.listdir(path):
        if file.endswith('.DS_Store'):
            os.remove(os.path.join(path, file))
                               
def update_to_huggingface():
    dataset = load_dataset("imagefolder", 
          data_dir=dataset_path, 
          split="train",
          features=Features({
              "image": DatasetImage(),
              "text": Value("string")
          }))
    print("\nDataset loaded!")
    dataset.push_to_hub(dataset_name, private=True)
    print('\nDataset uploaded!')
    