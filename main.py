import argparse
import logging
import random
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_argparse():
    """
    Sets up the argument parser for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Adds random noise to numerical data within a specified range.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("column_name", help="Name of the column to add noise to.")
    parser.add_argument("--min_noise", type=float, default=-0.1, help="Minimum noise value (as a fraction of the data value). Default: -0.1")
    parser.add_argument("--max_noise", type=float, default=0.1, help="Maximum noise value (as a fraction of the data value). Default: 0.1")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. Default: None")

    return parser.parse_args()


def add_noise(data, column_name, min_noise, max_noise, seed=None):
    """
    Adds random noise to the specified column in a pandas DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to add noise to.
        min_noise (float): The minimum noise value (as a fraction of the data value).
        max_noise (float): The maximum noise value (as a fraction of the data value).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with added noise.
    """

    try:
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the input file.")

        if not pd.api.types.is_numeric_dtype(data[column_name]):
            raise TypeError(f"Column '{column_name}' is not numeric.")

        if min_noise >= max_noise:
            raise ValueError("min_noise must be less than max_noise.")

        if seed is not None:
            np.random.seed(seed)  # Set seed for numpy as well
            random.seed(seed)       # Set seed for python's random

        data[column_name] = data[column_name].apply(
            lambda x: x * (1 + np.random.uniform(min_noise, max_noise))
        )

        return data

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except TypeError as e:
        logging.error(f"TypeError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


def main():
    """
    Main function to parse arguments, read data, add noise, and save the result.
    """
    args = setup_argparse()

    try:
        # Read the input CSV file
        logging.info(f"Reading input file: {args.input_file}")
        data = pd.read_csv(args.input_file)

        # Add noise to the specified column
        logging.info(f"Adding noise to column: {args.column_name}")
        data = add_noise(data, args.column_name, args.min_noise, args.max_noise, args.seed)

        # Save the modified DataFrame to a new CSV file
        logging.info(f"Saving output to: {args.output_file}")
        data.to_csv(args.output_file, index=False)

        logging.info("Data masking completed successfully.")

    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
    except pd.errors.EmptyDataError:
        logging.error(f"Input file is empty: {args.input_file}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage:
    # Create a sample CSV file named input.csv
    # with a column named 'salary'
    # To run the tool:
    # python main.py input.csv output.csv salary --min_noise -0.2 --max_noise 0.2 --seed 42

    #Example of using the tool with command line arguments.
    main()