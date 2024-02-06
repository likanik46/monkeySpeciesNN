import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

main_path = "C:/Users/PC/Desktop/archive/training/training"

subfolders = [f.path for f in os.scandir(main_path) if f.is_dir()]

columns = ["Label", "Latin Name", "Common Name", "Train Images", "Validation Images"]
df = pd.read_csv("C:/Users/PC/Desktop/archive/monkey_labels.txt", names=columns, skiprows=1)
df['Label'] = df['Label'].str.strip()
df['Latin Name'] = df['Latin Name'].replace("\t", "")
df['Latin Name'] = df['Latin Name'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()
df = df.set_index("Label")

monkeyNameDic = df["Common Name"]


folder_names = []
image_counts = []


num_folders = len(subfolders)
num_cols = 5  
num_rows = (num_folders + num_cols - 1) // num_cols
plt.figure(figsize=(15, 3*num_rows))


for i, folder in enumerate(subfolders, start=1):
    folder_name = os.path.basename(folder)
    num_images = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    folder_names.append(monkeyNameDic[folder_name])
    image_counts.append(num_images)
    

    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if image_files:
        img_path = os.path.join(folder, image_files[0])
        img = Image.open(img_path)
        
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img)
        plt.title(f"Slika {monkeyNameDic[folder_name]}", fontsize=8)  
        plt.axis('off')


plt.tight_layout(pad=3.0) 
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(folder_names, image_counts, color='skyblue')
plt.ylabel('Broj slika')
plt.title('Broj odbiraka svake klase')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
