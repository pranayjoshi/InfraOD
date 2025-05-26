import pandas as pd
import os
import shutil
from pre_process.my_utils import (
    open_file, reshape_dataframe, get_name, combine_labels,
    move_and_rename_image, rename_label, files_to_int,
    get_file_name, seq_to_int
)
from pre_process.config import (
    IMAGE_DIR, LABEL_DIR, ALL_IMAGES_DIR, EXTENSIONS,
    COLUMN_NAMES, IMAGE_WIDTH, IMAGE_HEIGHT
)

class DataProcessor:
    def __init__(self):
        self.all_labels = pd.DataFrame(columns=COLUMN_NAMES)
        
    def process_sequences(self):
        """Process all sequences and combine their labels"""
        count = 1
        sequences = os.listdir(IMAGE_DIR)
        
        for sequence in seq_to_int(sequences):
            # print("sequence", sequence)
            seq_dir = os.path.join(IMAGE_DIR, f"Sequence-{sequence}")
            images = os.listdir(seq_dir)
            filename = f"Seq{sequence}-IR.txt"
            
            # Process labels
            seq_label_df = open_file(filename, LABEL_DIR)
            seq_label_df = reshape_dataframe(seq_label_df)
            
            # Process images
            for image in files_to_int(images):
                # print("image", image)
                filename = get_file_name(image, sequence, EXTENSIONS)
                # print("filename", filename)
                rename_label(image, count, seq_label_df)
                move_and_rename_image(
                    os.path.join(seq_dir, filename),
                    os.path.join(ALL_IMAGES_DIR, f"{count}.jpg")
                )
                count += 1
            
            self.all_labels = combine_labels(self.all_labels, seq_label_df)
        
        return self.all_labels
    
    def process_labels(self, df):
        """Process and clean the labels dataframe"""
        # Convert types
        df['image_id'] = df['image_id'].astype(int)
        df['classes'] = df['classes'].astype(int)

        # Remap classes: 1 -> 0, 3 -> 1, others unchanged
        df.loc[df['classes'] == 1, 'classes'] = 0
        df.loc[df['classes'] == 3, 'classes'] = 1
        
        # Calculate center coordinates
        df['x_center'] = df['x'] + df['w']/2
        df['y_center'] = df['y'] + df['h']/2

        # Round all coordinate values to 2 decimal places
        for col in ['x', 'y', 'w', 'h', 'x_center', 'y_center']:
            df[col] = df[col].round(2)

        # Remove invalid entries
        df = df[df['w'] > 0]
        df = df[df['h'] > 0]

        print(len(df["classes"].unique()))
        print(df["classes"].unique())

        return df
    
    def save_labels(self, df, output_path):
        """Save processed labels to file"""
        df.to_csv(output_path, header=None, sep="\t", mode="w", index=None)
        print(f"Saved {len(df)} labels to {output_path}")
        print(f"Number of unique classes: {df['classes'].nunique()}")
        print("Class values:", df['classes'].unique()) 