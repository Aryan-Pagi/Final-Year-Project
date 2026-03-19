"""
Quick diagnostic to check all labels in the dataset
"""
import os

dataset_path = "dataset/raw_images"

if os.path.exists(dataset_path):
    label_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    print("\n" + "="*60)
    print("DATASET DIAGNOSTIC")
    print("="*60)
    print(f"Total labels: {len(label_dirs)}\n")
    
    for label in sorted(label_dirs):
        label_path = os.path.join(dataset_path, label)
        image_files = [f for f in os.listdir(label_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {label}: {len(image_files)} images")
    
    print("="*60 + "\n")
else:
    print(f"\nError: {dataset_path} not found\n")
