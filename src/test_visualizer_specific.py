import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pre_process.visualizer import Visualizer
from pre_process.data_processor import DataProcessor
from pre_process.config import ALL_IMAGES_DIR, LABEL_DIR

def test_specific_images():
    # Get DataFrame from DataProcessor
    processor = DataProcessor()
    df = processor.process_sequences()
    df = processor.process_labels(df)
    
    # Test each image
    for image_id in [1, 2, 3]:
        print(f"\nTesting visualization for image {image_id}")
        try:
            # Get image data
            image_path = os.path.join(ALL_IMAGES_DIR, f"{image_id}.jpg")
            x = np.array(Image.open(image_path), dtype=np.uint8)
            
            # Create figure and axes
            fig, ax = plt.subplots(1)
            
            # Display the image
            ax.imshow(x)
            
            # Get bounding box data for this image
            image_test = df[df["image_id"] == image_id]
            print(image_test)
            
            if len(image_test) > 0:
                # Calculate box coordinates
                x_min = image_test["x_center"].values - image_test["w"].values/2
                y_min = image_test["y_center"].values - image_test["h"].values/2
                
                # Create Rectangle patch
                rect = patches.Rectangle(
                    (x_min[0], y_min[0]),
                    image_test["w"].values[0],
                    image_test["h"].values[0],
                    linewidth=1,
                    edgecolor='r',
                    facecolor="none"
                )
                
                # Add the patch to the Axes
                ax.add_patch(rect)
                
                print(f"Found {len(image_test)} bounding boxes for image {image_id}")
                print("Bounding box coordinates:")
                for _, row in image_test.iterrows():
                    print(f"Class: {row['classes']}, Center: ({row['x_center']}, {row['y_center']}), Size: {row['w']}x{row['h']}")
            
            plt.title(f"Image {image_id} with Bounding Boxes")
            plt.show()
                
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            
    # Test class distribution
    print("\nTesting class distribution visualization")
    visualizer = Visualizer()
    visualizer.plot_class_distribution(df)

if __name__ == "__main__":
    test_specific_images() 