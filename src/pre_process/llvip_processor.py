import os
import shutil
import pandas as pd
import numpy as np
from pre_process.config import (
    CUR_DIR, DATASET_DIR, IMAGE_WIDTH, IMAGE_HEIGHT
)

class LLVIPProcessor:
    def __init__(self):
        self.llvip_dir = os.path.join(CUR_DIR, "LLVIP_data")
        self.train_dir = os.path.join(self.llvip_dir, "train")
        self.test_dir = os.path.join(self.llvip_dir, "test")
        
    def process_data(self):
        """Process LLVIP data and create train/val/test splits"""
        # Create dataset directory structure
        self._create_dataset_dirs()
        
        # Process train data
        train_images = self._get_image_files(self.train_dir)
        train_labels = self._get_label_files(self.train_dir)
        
        # Process test data
        test_images = self._get_image_files(self.test_dir)
        test_labels = self._get_label_files(self.test_dir)
        
        if not train_images and not test_images:
            print("Warning: No images found in LLVIP dataset")
            return
            
        print(f"Found {len(train_images)} training images and {len(test_images)} test images")
        
        # Combine all data
        all_images = train_images + test_images
        all_labels = train_labels + test_labels
        
        # Shuffle data
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Calculate split sizes
        n_samples = len(all_images)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        # Split data
        train_images = all_images[:train_size]
        train_labels = all_labels[:train_size]
        
        val_images = all_images[train_size:train_size + val_size]
        val_labels = all_labels[train_size:train_size + val_size]
        
        test_images = all_images[train_size + val_size:]
        test_labels = all_labels[train_size + val_size:]
        
        # Process each split
        self._process_split(train_images, train_labels, "train")
        self._process_split(val_images, val_labels, "val")
        self._process_split(test_images, test_labels, "test")
        
        print(f"Processed {len(train_images)} training images")
        print(f"Processed {len(val_images)} validation images")
        print(f"Processed {len(test_images)} test images")
    
    def _create_dataset_dirs(self):
        """Create necessary directories for the dataset"""
        for split in ["train", "val", "test"]:
            for dir_type in ["images", "labels"]:
                path = os.path.join(DATASET_DIR, dir_type, split)
                if not os.path.exists(path):
                    os.makedirs(path)
    
    def _get_image_files(self, directory):
        """Get list of image files from directory"""
        image_dir = os.path.join(directory, "images")
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory not found: {image_dir}")
            return []
        return [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    def _get_label_files(self, directory):
        """Get list of label files from directory"""
        label_dir = os.path.join(directory, "labels")
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found: {label_dir}")
            return []
        return [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    def _process_split(self, image_files, label_files, split_name):
        """Process a single split (train/val/test)"""
        for img_file, label_file in zip(image_files, label_files):
            try:
                # Determine source directory based on where the image exists
                src_dir = self.train_dir if os.path.exists(os.path.join(self.train_dir, "images", img_file)) else self.test_dir
                
                # Copy image
                src_img = os.path.join(src_dir, "images", img_file)
                dst_img = os.path.join(DATASET_DIR, "images", split_name, img_file)
                
                if not os.path.exists(src_img):
                    print(f"Warning: Source image not found: {src_img}")
                    continue
                    
                shutil.copy2(src_img, dst_img)
                
                # Process and copy label
                src_label = os.path.join(src_dir, "labels", label_file)
                dst_label = os.path.join(DATASET_DIR, "labels", split_name, label_file)
                
                if not os.path.exists(src_label):
                    print(f"Warning: Source label not found: {src_label}")
                    continue
                
                # Read and process label
                with open(src_label, 'r') as f:
                    lines = f.readlines()
                
                # Write processed label
                with open(dst_label, 'w') as f:
                    for line in lines:
                        # LLVIP labels are already in YOLO format
                        f.write(line)
                        
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue 