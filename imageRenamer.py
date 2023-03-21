import os
import json

# Open the JSON file and read the contents
with open('birds.json', 'r') as f:
    data = json.load(f)
    

# Iterate over the folders in the JSON file
for categories in data['categories']:
    # Get the current name and the new name from the JSON file
    current_name = categories['image_dir_name']
    new_name = categories['common_name']

    # Use the os.listdir() function to get a list of all the directories in the current path
    for dir in os.listdir('train'):
        # If the current directory matches the current name, rename it
        if dir == current_name:
            try:
                os.rename(f'train/{dir}', f'train/{new_name}')
            except OSError as e:
                print(f"Error renaming {current_name}: {e}")