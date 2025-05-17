# src/load/file_loader.py

import pandas as pd
import logging
import os
from typing import Optional, Dict, Any, Union

# Get a logger for this module
logger = logging.getLogger(__name__)

class FileLoader:
    """
    A class to save pandas DataFrames to various file formats.
    """

    def __init__(self):
        """
        Initializes the FileLoader.
        """
        logger.info("FileLoader initialized.")

    def _ensure_directory_exists(self, file_path: str) -> None:
        """
        Ensures that the directory for the given file_path exists.
        If not, it attempts to create it.
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)
                raise # Re-raise the exception as this is critical

    def save_to_csv(self,
                    df: pd.DataFrame,
                    file_path: str,
                    index: bool = False,
                    header: bool = True,
                    sep: str = ',',
                    encoding: Optional[str] = 'utf-8',
                    **kwargs) -> bool:
        """
        Saves a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): The full path to the output CSV file.
            index (bool): Whether to write DataFrame index as a column.
            header (bool): Whether to write out the column names.
            sep (str): Delimiter to use.
            encoding (Optional[str]): Encoding to use for the output file.
            **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_csv().

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame for CSV saving.")
            return False
        
        logger.info(f"Attempting to save DataFrame to CSV: {file_path}")
        try:
            self._ensure_directory_exists(file_path)
            df.to_csv(file_path, index=index, header=header, sep=sep, encoding=encoding, **kwargs)
            logger.info(f"Successfully saved DataFrame to CSV: {file_path} ({len(df)} rows)")
            return True
        except Exception as e:
            logger.error(f"Error saving DataFrame to CSV {file_path}: {e}", exc_info=True)
            return False

    def save_to_json(self,
                     df: pd.DataFrame,
                     file_path: str,
                     orient: str = 'records', # Common options: 'records', 'split', 'index', 'columns', 'values', 'table'
                     lines: bool = False, # If True, writes JSON Lines format (one JSON object per line)
                     indent: Optional[int] = None, # Indent level for pretty-printing
                     date_format: Optional[str] = 'iso', # Format for datetime objects
                     **kwargs) -> bool:
        """
        Saves a DataFrame to a JSON file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): The full path to the output JSON file.
            orient (str): Expected JSON string format.
            lines (bool): If True, each row will be a separate JSON object on a new line.
            indent (Optional[int]): If not None, JSON will be pretty-printed with this indent level.
            date_format (Optional[str]): Type of date conversion, 'iso' or 'epoch'.
            **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_json().

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame for JSON saving.")
            return False

        logger.info(f"Attempting to save DataFrame to JSON: {file_path} (orient='{orient}', lines={lines})")
        try:
            self._ensure_directory_exists(file_path)
            df.to_json(file_path, orient=orient, lines=lines, indent=indent, date_format=date_format, **kwargs)
            logger.info(f"Successfully saved DataFrame to JSON: {file_path} ({len(df)} rows)")
            return True
        except Exception as e:
            logger.error(f"Error saving DataFrame to JSON {file_path}: {e}", exc_info=True)
            return False

    def save_to_parquet(self,
                        df: pd.DataFrame,
                        file_path: str,
                        engine: str = 'auto', # 'auto', 'pyarrow', 'fastparquet'
                        compression: Optional[str] = 'snappy', # 'snappy', 'gzip', 'brotli', None
                        index: Optional[bool] = None, # If True, include index in output. Default depends on engine.
                        **kwargs) -> bool:
        """
        Saves a DataFrame to a Parquet file.
        Requires 'pyarrow' or 'fastparquet' library to be installed.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): The full path to the output Parquet file.
            engine (str): Parquet library to use.
            compression (Optional[str]): Compression codec to use.
            index (Optional[bool]): If True, store the DataFrame index in the Parquet file.
            **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_parquet().

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame for Parquet saving.")
            return False

        logger.info(f"Attempting to save DataFrame to Parquet: {file_path} (engine='{engine}', compression='{compression}')")
        try:
            self._ensure_directory_exists(file_path)
            df.to_parquet(file_path, engine=engine, compression=compression, index=index, **kwargs)
            logger.info(f"Successfully saved DataFrame to Parquet: {file_path} ({len(df)} rows)")
            return True
        except ImportError:
            logger.error(f"Failed to save to Parquet: '{engine if engine != 'auto' else 'pyarrow or fastparquet'}' library not found. Please install it (e.g., pip install pyarrow).")
            return False
        except Exception as e:
            logger.error(f"Error saving DataFrame to Parquet {file_path}: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sample DataFrame for testing
    data = {
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'Timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-02 11:30:00', '2025-01-03 12:15:00'])
    }
    sample_df = pd.DataFrame(data)

    loader = FileLoader()

    # Define output directory for tests
    output_dir = "test_output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test CSV saving
    csv_file_path = os.path.join(output_dir, "sample_output.csv")
    logger.info(f"\n--- Testing CSV Saving to {csv_file_path} ---")
    success_csv = loader.save_to_csv(sample_df, csv_file_path)
    logger.info(f"CSV save successful: {success_csv}")
    if success_csv and os.path.exists(csv_file_path):
        logger.info(f"CSV file created. Content:\n{pd.read_csv(csv_file_path)}")


    # Test JSON saving (records orient)
    json_file_path_records = os.path.join(output_dir, "sample_output_records.json")
    logger.info(f"\n--- Testing JSON Saving (records) to {json_file_path_records} ---")
    success_json_records = loader.save_to_json(sample_df, json_file_path_records, orient='records', indent=2)
    logger.info(f"JSON (records) save successful: {success_json_records}")
    if success_json_records and os.path.exists(json_file_path_records):
         with open(json_file_path_records, 'r') as f:
            logger.info(f"JSON (records) file created. Content:\n{f.read(500)}...") # Print first 500 chars

    # Test JSON saving (JSON Lines format)
    jsonl_file_path = os.path.join(output_dir, "sample_output.jsonl")
    logger.info(f"\n--- Testing JSON Lines Saving to {jsonl_file_path} ---")
    success_jsonl = loader.save_to_json(sample_df, jsonl_file_path, orient='records', lines=True)
    logger.info(f"JSON Lines save successful: {success_jsonl}")
    if success_jsonl and os.path.exists(jsonl_file_path):
        with open(jsonl_file_path, 'r') as f:
            logger.info(f"JSON Lines file created. Content:\n{f.read(500)}...")

    # Test Parquet saving
    # Note: This requires pyarrow or fastparquet to be installed.
    # If not installed, it will log an error and return False.
    parquet_file_path = os.path.join(output_dir, "sample_output.parquet")
    logger.info(f"\n--- Testing Parquet Saving to {parquet_file_path} ---")
    try:
        import pyarrow # or fastparquet
        success_parquet = loader.save_to_parquet(sample_df, parquet_file_path)
        logger.info(f"Parquet save successful: {success_parquet}")
        if success_parquet and os.path.exists(parquet_file_path):
            logger.info(f"Parquet file created. Reading back for verification:\n{pd.read_parquet(parquet_file_path)}")
    except ImportError:
        logger.warning("Skipping Parquet save test as 'pyarrow' or 'fastparquet' is not installed.")


    # Test saving to a non-existent nested directory
    nested_csv_path = os.path.join(output_dir, "nested_dir", "another_sample.csv")
    logger.info(f"\n--- Testing CSV Saving to nested directory {nested_csv_path} ---")
    success_nested_csv = loader.save_to_csv(sample_df, nested_csv_path)
    logger.info(f"Nested CSV save successful: {success_nested_csv}")
    if success_nested_csv and os.path.exists(nested_csv_path):
        logger.info(f"Nested CSV file created at {nested_csv_path}")


    # Clean up test output directory (optional)
    # import shutil
    # if os.path.exists(output_dir):
    #     logger.info(f"\nCleaning up test output directory: {output_dir}")
    #     shutil.rmtree(output_dir)
    #     logger.info("Test output directory removed.")

    logger.info("\nFileLoader tests finished.")
