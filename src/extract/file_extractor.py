# src/extract/file_extractor.py

import pandas as pd
import logging
import os
import json
from typing import Optional, List, Dict, Any, Union

# Get a logger for this module
logger = logging.getLogger(__name__) # This will use the 'financial_etl_pipeline' logger if called from main
                                     # or a logger named 'src.extract.file_extractor' if run standalone
                                     # after logging is configured.

class FileExtractor:
    """
    A class to extract data from various file types.
    """

    def __init__(self, file_path: str):
        """
        Initializes the FileExtractor with the path to the file.

        Args:
            file_path (str): The full path to the file to be extracted.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found at path: {file_path}")
            raise FileNotFoundError(f"File not found at path: {file_path}")
        self.file_path = file_path
        self.file_type = self._get_file_type()
        logger.info(f"FileExtractor initialized for file: {self.file_path} (type: {self.file_type})")

    def _get_file_type(self) -> Optional[str]:
        """
        Determines the file type based on its extension.

        Returns:
            Optional[str]: The file extension (e.g., 'csv', 'json') or None if not determined.
        """
        try:
            return self.file_path.split('.')[-1].lower()
        except IndexError:
            logger.warning(f"Could not determine file type for: {self.file_path} (no extension)")
            return None

    def extract_data(self, chunk_size: Optional[int] = None, **kwargs) -> Optional[Union[pd.DataFrame, List[pd.DataFrame], List[Dict[Any, Any]]]]:
        """
        Extracts data from the file based on its determined type.

        Args:
            chunk_size (Optional[int]): For CSV files, number of rows to read per chunk.
                                         If None, reads the entire file at once.
            **kwargs: Additional keyword arguments to pass to the pandas read function.

        Returns:
            Optional[Union[pd.DataFrame, List[pd.DataFrame], List[Dict[Any, Any]]]]:
            A pandas DataFrame (or list of DataFrames if chunking) for CSV,
            a list of dictionaries for JSON, or None if extraction fails or file type is unsupported.
        """
        logger.info(f"Attempting to extract data from {self.file_path} as {self.file_type}")
        if self.file_type == 'csv':
            return self._extract_csv(chunk_size=chunk_size, **kwargs)
        elif self.file_type == 'json':
            return self._extract_json(**kwargs)
        # Add more file types here
        # elif self.file_type == 'parquet':
        #     return self._extract_parquet(**kwargs)
        # elif self.file_type == 'xlsx' or self.file_type == 'xls':
        #     return self._extract_excel(**kwargs)
        else:
            logger.error(f"Unsupported file type: {self.file_type} for file {self.file_path}")
            return None

    def _extract_csv(self, chunk_size: Optional[int] = None, **kwargs) -> Optional[Union[pd.DataFrame, List[pd.DataFrame]]]:
        """
        Extracts data from a CSV file.

        Args:
            chunk_size (Optional[int]): Number of rows to read per chunk.
                                         If None, reads the entire file at once.
            **kwargs: Additional keyword arguments to pass to pandas.read_csv().

        Returns:
            Optional[Union[pd.DataFrame, List[pd.DataFrame]]]: A DataFrame or list of DataFrames, or None on error.
        """
        try:
            if chunk_size:
                logger.info(f"Reading CSV file in chunks of {chunk_size} rows: {self.file_path}")
                chunks = []
                for chunk in pd.read_csv(self.file_path, chunksize=chunk_size, **kwargs):
                    logger.debug(f"Read chunk of size {len(chunk)} from {self.file_path}")
                    chunks.append(chunk)
                if not chunks:
                    logger.warning(f"CSV file is empty or resulted in no chunks: {self.file_path}")
                    return pd.DataFrame() # Return empty DataFrame for consistency
                logger.info(f"Successfully extracted {len(chunks)} chunks from CSV: {self.file_path}")
                return chunks
            else:
                logger.info(f"Reading entire CSV file: {self.file_path}")
                df = pd.read_csv(self.file_path, **kwargs)
                logger.info(f"Successfully extracted {len(df)} rows from CSV: {self.file_path}")
                return df
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.file_path}")
            # This case is already handled in __init__, but good for defense in depth
            raise
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file is empty: {self.file_path}")
            return pd.DataFrame() # Return an empty DataFrame
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file {self.file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting CSV {self.file_path}: {e}")
            return None

    def _extract_json(self, **kwargs) -> Optional[List[Dict[Any, Any]]]:
        """
        Extracts data from a JSON file.
        Assumes the JSON file contains a list of objects or a single object.

        Args:
            **kwargs: Additional keyword arguments (e.g., 'orient' if using pandas.read_json).
                      Currently, this method uses the standard json library.

        Returns:
            Optional[List[Dict[Any, Any]]]: A list of dictionaries, or None on error.
                                            Could also return a DataFrame if pandas.read_json is preferred.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully extracted data from JSON file: {self.file_path}")
            if isinstance(data, list):
                return data
            elif isinstance(data, dict): # If it's a single JSON object, wrap it in a list
                return [data]
            else:
                logger.warning(f"JSON data in {self.file_path} is not a list or a dictionary. Type: {type(data)}")
                return None
        except FileNotFoundError:
            logger.error(f"JSON file not found: {self.file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file {self.file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting JSON {self.file_path}: {e}")
            return None

    # --- Placeholder for other file types ---
    # def _extract_parquet(self, **kwargs) -> Optional[pd.DataFrame]:
    #     """Extracts data from a Parquet file."""
    #     try:
    #         logger.info(f"Reading Parquet file: {self.file_path}")
    #         df = pd.read_parquet(self.file_path, **kwargs)
    #         logger.info(f"Successfully extracted {len(df)} rows from Parquet: {self.file_path}")
    #         return df
    #     except Exception as e:
    #         logger.error(f"Error extracting Parquet file {self.file_path}: {e}")
    #         return None

    # def _extract_excel(self, sheet_name: Union[str, int, None] = 0, **kwargs) -> Optional[pd.DataFrame]:
    #     """Extracts data from an Excel file."""
    #     try:
    #         logger.info(f"Reading Excel file: {self.file_path}, sheet: {sheet_name}")
    #         df = pd.read_excel(self.file_path, sheet_name=sheet_name, **kwargs)
    #         logger.info(f"Successfully extracted {len(df)} rows from Excel sheet '{sheet_name}': {self.file_path}")
    #         return df
    #     except Exception as e:
    #         logger.error(f"Error extracting Excel file {self.file_path}: {e}")
    #         return None

if __name__ == '__main__':
    # This is for basic testing of the FileExtractor.
    # In a real application, logging would be configured by the main script.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy files for testing
    dummy_csv_path = 'dummy_transactions.csv'
    dummy_json_path = 'dummy_transactions.json'
    dummy_empty_csv_path = 'empty.csv'
    dummy_malformed_csv_path = 'malformed.csv'

    # Create dummy CSV
    with open(dummy_csv_path, 'w') as f:
        f.write("TransactionID,Amount,Currency\n")
        f.write("TXN001,100.50,USD\n")
        f.write("TXN002,75.20,EUR\n")
        f.write("TXN003,200.00,USD\n")

    # Create dummy JSON (list of objects)
    dummy_json_data = [
        {"TransactionID": "TXN004", "Amount": 50.00, "Currency": "GBP"},
        {"TransactionID": "TXN005", "Amount": 120.75, "Currency": "USD"}
    ]
    with open(dummy_json_path, 'w') as f:
        json.dump(dummy_json_data, f)

    # Create empty CSV
    with open(dummy_empty_csv_path, 'w') as f:
        f.write("") # or "Header1,Header2\n"

    # Create malformed CSV
    with open(dummy_malformed_csv_path, 'w') as f:
        f.write("ID,Value\n")
        f.write("1,SomeValue\"unclosed_quote\n") # Malformed line

    logger.info("-" * 30)
    logger.info("Testing CSV Extractor:")
    csv_extractor = FileExtractor(dummy_csv_path)
    csv_data = csv_extractor.extract_data()
    if csv_data is not None:
        print("CSV Data:")
        print(csv_data.head())

    logger.info("-" * 30)
    logger.info("Testing CSV Extractor with Chunks:")
    csv_extractor_chunked = FileExtractor(dummy_csv_path)
    csv_chunks = csv_extractor_chunked.extract_data(chunk_size=1)
    if csv_chunks is not None and isinstance(csv_chunks, list):
        print(f"Number of CSV chunks: {len(csv_chunks)}")
        for i, chunk_df in enumerate(csv_chunks):
            print(f"Chunk {i+1}:")
            print(chunk_df)

    logger.info("-" * 30)
    logger.info("Testing JSON Extractor:")
    json_extractor = FileExtractor(dummy_json_path)
    json_data = json_extractor.extract_data()
    if json_data is not None:
        print("\nJSON Data:")
        # If using pandas.read_json, this would be a DataFrame
        # For current json.load, it's a list of dicts
        for record in json_data:
            print(record)

    logger.info("-" * 30)
    logger.info("Testing Empty CSV Extractor:")
    empty_csv_extractor = FileExtractor(dummy_empty_csv_path)
    empty_csv_data = empty_csv_extractor.extract_data()
    if empty_csv_data is not None:
        print("\nEmpty CSV Data (should be an empty DataFrame):")
        print(empty_csv_data)

    logger.info("-" * 30)
    logger.info("Testing Malformed CSV Extractor:")
    malformed_csv_extractor = FileExtractor(dummy_malformed_csv_path)
    malformed_csv_data = malformed_csv_extractor.extract_data() # Should log an error and return None
    if malformed_csv_data is None:
        print("\nMalformed CSV Data: Extraction correctly returned None (error logged).")

    logger.info("-" * 30)
    logger.info("Testing Non-Existent File:")
    try:
        non_existent_extractor = FileExtractor("non_existent_file.txt")
    except FileNotFoundError as e:
        print(f"\nSuccessfully caught error for non-existent file: {e}")

    # Clean up dummy files
    os.remove(dummy_csv_path)
    os.remove(dummy_json_path)
    os.remove(dummy_empty_csv_path)
    os.remove(dummy_malformed_csv_path)
    logger.info("Cleaned up dummy files.")
