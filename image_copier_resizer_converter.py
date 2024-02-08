# An image resizer function
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_resizer(image_path, output_file, resize):
    basename = os.path.basename(image_path)
    outpath = os.path.join(output_file, basename)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)

# Create the directory to save resized images
os.makedirs('resized_images', exist_ok=True)

# Deploy the function in a for loop
files = os.listdir('HAM10000_images_part1&2')
for file in files:
    file_path = os.path.join('HAM10000_images_part1&2', file)
    image_resizer(file_path, 'resized_images', [224, 224])


    
    
    
#Another resizer that skips image that has already been copied    
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_resizer(image_path, output_file, resize):
    basename = os.path.basename(image_path)
    outpath = os.path.join(output_file, basename)

    # Check if the resized image already exists
    if os.path.exists(outpath):
        print(f"Skipped: {basename} already resized.")
        return

    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)

# Deploy the function in a for loop
files = os.listdir(r'JustRAIGS_Train')
for file in files:
    file_path = os.path.join(r'JustRAIGS_Train', file)
    image_resizer(file_path, 'resized_images', [224, 224])

    
    
    
# An image copier function
import shutil
import pandas as pd

df = pd.read_csv('train.csv')

def image_copier(source_folder, destination_folder):
    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Iterate over each file in the source folder
    for file in files:
        # Get the basename of the file without the extension
        filename = os.path.splitext(file)[0]

        if filename in df['image_id'].values:
            # Create the source and destination paths
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            # Copy the file to the destination folder
            shutil.copyfile(source_path, destination_path)
            print(f"Image {file} copied successfully.")
            


# Some images are in png format and their names are not matching with the ones in the JustRAIGS_Train_labels.csv
# Let's create a function to save them as jpeg (for reduced sizes) and rename them
from PIL import Image
import os

def convert_images_to_jpg(input_folder):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpeg')):
            # Check if the image is already in JPEG format
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                # Open the image
                img = Image.open(input_path)

                # Convert to RGB mode (remove alpha channel)
                img = img.convert('RGB')

                # Save as JPEG with uppercase extension
                new_filename = os.path.splitext(filename)[0] + '.JPG'
                output_path = os.path.join(input_folder, new_filename)
                img.save(output_path, 'JPEG')

                print(f"Converted: {filename} to {new_filename}")
                os.remove(input_path)  # Remove the original PNG or JPEG file
            else:
                # Rename images with '.jpeg' extension to '.JPG'
                new_filename = os.path.splitext(filename)[0] + '.JPG'
                new_path = os.path.join(input_folder, new_filename)
                os.rename(input_path, new_path)
                print(f"Renamed: {filename} to {new_filename}")

# Now call the function on 'train_images'
input_folder = 'C:\\Users\\USER\\Justraigs\\train_images'
convert_images_to_jpg(input_folder)