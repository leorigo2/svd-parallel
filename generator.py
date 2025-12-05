import numpy as np
import random
import os

## THANKS TO GEMINI3 FOR THIS BORING WORK

# Define the output file path and the total number of matrices to generate
OUTPUT_FILE = "dataset.txt"
NUM_MATRICES = 1

def generate_and_save_dataset():
    """
    Generates a dataset of random matrices and saves them to a text file.
    
    The file format is designed for easy C parsing:
    1. Line 1: Total number of matrices (NUM_MATRICES).
    2. For each matrix:
        a. Line: Number of rows (R) and number of columns (C), space-separated.
        b. R lines follow, each containing C space-separated floating-point values.
    """
    
    print(f"Starting dataset generation. Generating {NUM_MATRICES} matrices...")

    try:
        with open(OUTPUT_FILE, 'w') as f:
            # 1. Write the total number of matrices on the first line
            f.write(f"{NUM_MATRICES}\n")
            print(f"Total matrices: {NUM_MATRICES}")
            
            for i in range(NUM_MATRICES):
                # Randomly determine dimensions (e.g., between 2x2 and 10x10)
                R = random.randint(1000, 1000)
                C = random.randint(1000, 1000)
                
                # Generate a random matrix with real (float) values
                # Values will be between -10.0 and 10.0 for variety
                matrix = np.random.uniform(-10.0, 10.0, (R, C))
                
                # 2a. Write the dimensions (R C)
                f.write(f"{R} {C}\n")
                
                # 2b. Write the matrix data, row by row
                for row in matrix:
                    # np.savetxt is excellent for formatting float output clearly
                    # We use '%.6f' to ensure consistent precision for C parsing
                    row_str = " ".join(f"{x:.2f}" for x in row)
                    f.write(f"{row_str}\n")
                    
                print(f"  - Generated matrix {i+1}/{NUM_MATRICES}: {R} rows x {C} columns")

        print(f"\nSuccessfully generated dataset and saved to '{OUTPUT_FILE}'.")
        print("This file is now ready to be read by a C program.")

    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

if __name__ == "__main__":
    generate_and_save_dataset()