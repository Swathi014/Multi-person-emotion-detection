import os
import shutil
import random

def create_balanced_split():
    # Your current messy training folder with the ~49k images
    source_dir = r'./Emotions_Datasets/train' 
    
    # The new, perfectly clean directory we are about to build
    output_dir = r'./Balanced_Dataset'
    train_out = os.path.join(output_dir, 'train')
    test_out = os.path.join(output_dir, 'test')

    if not os.path.exists(source_dir):
        print(f"Error: Could not find the source directory at {source_dir}")
        return

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # 1. Find the smallest class to balance everything
    class_counts = {}
    for cls in classes:
        images = os.listdir(os.path.join(source_dir, cls))
        class_counts[cls] = len(images)
    
    min_images = min(class_counts.values())
    print(f"--- Data Diagnostics ---")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")
    print(f"\nBalancing all classes to exactly {min_images} images each...\n")

    # 2. Setup the split ratio
    train_ratio = 0.80
    train_count = int(min_images * train_ratio)
    test_count = min_images - train_count

    # 3. Process each class
    for cls in classes:
        # Create the new destination folders safely
        os.makedirs(os.path.join(train_out, cls), exist_ok=True)
        os.makedirs(os.path.join(test_out, cls), exist_ok=True)

        images = os.listdir(os.path.join(source_dir, cls))
        random.shuffle(images) # Shuffle so we get a random variety
        
        # Take only the amount needed for perfect balance
        selected_images = images[:min_images]
        
        train_images = selected_images[:train_count]
        test_images = selected_images[train_count:]

        # Copy over the train images
        for img in train_images:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(train_out, cls, img)
            shutil.copy(src, dst)
            
        # Copy over the test images
        for img in test_images:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(test_out, cls, img)
            shutil.copy(src, dst)

        print(f"Finished formatting '{cls}' -> Train: {len(train_images)} | Test: {len(test_images)}")

    print(f"\nSuccess! Your clean data is ready inside the '{output_dir}' folder.")

if __name__ == '__main__':
    create_balanced_split()