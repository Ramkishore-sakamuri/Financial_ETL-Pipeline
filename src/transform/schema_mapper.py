# src/transform/schema_mapper.py

import pandas as pd
import logging
from typing import List, Dict, Any, Union

# Get a logger for this module
logger = logging.getLogger(__name__)

class SchemaMapper:
    """
    A class to map a DataFrame to a target schema, including column selection,
    renaming, reordering, and type casting.
    """

    def __init__(self, target_schema: List[Dict[str, Any]]):
        """
        Initializes the SchemaMapper with the target schema definition.

        Args:
            target_schema (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary defines a target column. Each dictionary should have:
                - 'target_name' (str): The desired name of the column in the output.
                - 'source_column' (str): The name of the column in the input DataFrame.
                - 'dtype' (Union[str, type]): The desired pandas/numpy data type for the column
                                              (e.g., 'str', 'int64', 'float64', 'datetime64[ns]', bool).
                - 'required' (bool, optional): If True, an error will be logged if the source column
                                               is missing. Defaults to False (column will be created with NaNs).
        """
        if not isinstance(target_schema, list) or not all(isinstance(item, dict) for item in target_schema):
            raise ValueError("target_schema must be a list of dictionaries.")
        
        required_keys = {'target_name', 'source_column', 'dtype'}
        for i, col_def in enumerate(target_schema):
            if not required_keys.issubset(col_def.keys()):
                raise ValueError(
                    f"Each item in target_schema must contain keys: {required_keys}. "
                    f"Error in item {i}: {col_def}"
                )
        
        self.target_schema = target_schema
        logger.info(f"SchemaMapper initialized with target schema for {len(target_schema)} columns.")

    def map_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps the input DataFrame to the target schema.

        Args:
            df (pd.DataFrame): The input DataFrame to be mapped.

        Returns:
            pd.DataFrame: A new DataFrame conforming to the target schema.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input to map_df must be a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        logger.info(f"Starting schema mapping for DataFrame with shape {df.shape}.")
        
        # Create a new DataFrame to hold the results, ensuring order
        output_df = pd.DataFrame()
        final_column_order = [col_def['target_name'] for col_def in self.target_schema]

        for col_def in self.target_schema:
            target_name = col_def['target_name']
            source_column = col_def['source_column']
            target_dtype = col_def['dtype']
            is_required = col_def.get('required', False) # Default to False if not specified

            logger.debug(f"Processing target column: '{target_name}' from source: '{source_column}' with dtype: '{target_dtype}'.")

            if source_column in df.columns:
                output_df[target_name] = df[source_column]
                
                # Attempt type conversion
                try:
                    if output_df[target_name].isnull().all() and str(target_dtype).startswith('datetime'):
                        # If all values are null, astype to datetime64[ns] might fail or be ambiguous
                        # pd.to_datetime handles this better for all-NaN series
                        output_df[target_name] = pd.to_datetime(output_df[target_name], errors='coerce')
                    elif str(target_dtype).startswith('datetime'):
                         output_df[target_name] = pd.to_datetime(output_df[target_name], errors='coerce')
                    elif target_dtype == bool or target_dtype == 'bool':
                        # Handle boolean conversion more carefully, e.g., map common strings
                        # For simplicity here, direct astype. In practice, more robust bool conversion is needed.
                        # Example: map {'true', '1', 'yes'} to True, {'false', '0', 'no'} to False
                        if output_df[target_name].dtype == 'object': # if it's string like 'True', 'False'
                             bool_map = {'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False}
                             # Convert to lower string to handle 'True'/'true'
                             output_df[target_name] = output_df[target_name].astype(str).str.lower().map(bool_map)
                        output_df[target_name] = output_df[target_name].astype(bool)

                    else:
                        output_df[target_name] = output_df[target_name].astype(target_dtype)
                    logger.debug(f"Successfully cast column '{target_name}' to '{target_dtype}'.")
                except Exception as e:
                    logger.error(f"Error casting column '{target_name}' (from '{source_column}') to type '{target_dtype}': {e}. Column will be kept as is or may contain NaNs.", exc_info=True)
                    # If casting fails, the column remains as is, or pandas might have coerced some values.
                    # For critical type safety, you might want to fill with NaNs or raise an error.
                    # output_df[target_name] = pd.Series(index=output_df.index, dtype=object) # Example: fill with NaNs
            else:
                log_message = f"Source column '{source_column}' for target '{target_name}' not found in input DataFrame."
                if is_required:
                    logger.error(log_message + " This column is marked as required.")
                    # Depending on policy, you might raise an error or create an empty column
                    # For now, create an empty column of the target type if possible
                else:
                    logger.warning(log_message + " Creating empty column.")
                
                # Create an empty series of the target type
                try:
                    # For some dtypes (like custom ones or complex ones), creating an empty series might need care
                    if str(target_dtype).startswith('datetime'):
                        output_df[target_name] = pd.Series(pd.NaT, index=df.index, dtype=target_dtype)
                    else:
                        output_df[target_name] = pd.Series(dtype=target_dtype, index=df.index)
                except Exception as e_dtype:
                    logger.warning(f"Could not create empty column '{target_name}' with dtype '{target_dtype}': {e_dtype}. Using object type.")
                    output_df[target_name] = pd.Series(dtype=object, index=df.index)
        
        # Ensure final column order and drop any extra columns that might have been created if source_column was a typo of target_name
        # and df had a column with target_name already.
        # The current loop structure should prevent this, but reindexing is a safe final step.
        output_df = output_df.reindex(columns=final_column_order)

        logger.info(f"Schema mapping complete. Output DataFrame shape: {output_df.shape}, Columns: {output_df.columns.tolist()}")
        return output_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sample input DataFrame (could be output from TransactionTransformer)
    input_data = {
        'TxnID': ['T001', 'T002', 'T003', 'T004'],
        'Timestamp': ['2025-05-01T08:30:15', '2025-05-01T09:15:45', None, '2025-05-02T11:30:00'],
        'AccountNo': ['ACC001', 'ACC002', 'ACC001', 'ACC003'],
        'Type': ['DEBIT', 'CREDIT', 'DEBIT', 'DEBIT'],
        'OriginalAmount': [55.75, 1500.00, 89.99, 22.40],
        'OriginalCurrency': ['USD', 'USD', 'EUR', 'USD'],
        'AmountUSD': [55.75, 1500.00, 97.19, 22.40], # Assuming EUR was converted
        'IsIntlTxn': [False, False, True, False],
        'Details': ['Groceries', 'Salary', 'Books EU', 'Coffee'],
        'StandardCategory': ['GROCERIES', 'INCOME', 'RETAIL_SHOPPING', 'FOOD_BEVERAGE'],
        'AutoCategory': ['SHOPPING', 'FINANCE', 'SHOPPING', 'LEISURE'],
        'TxnStatus': ['COMPLETED', 'COMPLETED', 'COMPLETED', 'PENDING'],
        'ExtraColumn': ['extra1', 'extra2', 'extra3', 'extra4'] # This should be dropped
    }
    input_df = pd.DataFrame(input_data)
    # Manually set Timestamp to datetime to simulate previous step
    input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp'])
    input_df['IsIntlTxn'] = input_df['IsIntlTxn'].astype(bool)


    logger.info("Original DataFrame for SchemaMapper:")
    print(input_df)
    logger.info(f"Original dtypes:\n{input_df.dtypes}")

    # Define the target schema
    target_schema_definition = [
        {'target_name': 'TransactionIdentifier', 'source_column': 'TxnID', 'dtype': 'str', 'required': True},
        {'target_name': 'ExecutionTime', 'source_column': 'Timestamp', 'dtype': 'datetime64[ns]'},
        {'target_name': 'AccountNumber', 'source_column': 'AccountNo', 'dtype': 'str'},
        {'target_name': 'TransactionValue', 'source_column': 'AmountUSD', 'dtype': 'float64'},
        {'target_name': 'TransactionCurrency', 'source_column': 'OriginalCurrency', 'dtype': 'str'},
        {'target_name': 'PrimaryCategory', 'source_column': 'StandardCategory', 'dtype': 'str'},
        {'target_name': 'SecondaryCategory', 'source_column': 'AutoCategory', 'dtype': 'str'},
        {'target_name': 'Status', 'source_column': 'TxnStatus', 'dtype': 'category'}, # Example of category type
        {'target_name': 'IsOverseas', 'source_column': 'IsIntlTxn', 'dtype': 'bool'},
        {'target_name': 'Notes', 'source_column': 'Details', 'dtype': 'str'},
        {'target_name': 'SourceSystemID', 'source_column': 'SystemID_NonExistent', 'dtype': 'str', 'required': False}, # Test missing non-required
        {'target_name': 'CriticalFlag', 'source_column': 'CriticalCol_NonExistent', 'dtype': 'bool', 'required': True}, # Test missing required
    ]

    schema_mapper = SchemaMapper(target_schema=target_schema_definition)
    mapped_df = schema_mapper.map_df(input_df.copy()) # Pass a copy

    logger.info("\nMapped DataFrame:")
    print(mapped_df)
    logger.info(f"\nMapped Dtypes:\n{mapped_df.dtypes}")
    logger.info(f"\nMapped Shape: {mapped_df.shape}")
    logger.info(f"\nMapped Columns: {mapped_df.columns.tolist()}")

    # Test with an input that is not a DataFrame
    logger.info("\n--- Testing with invalid input type ---")
    try:
        schema_mapper.map_df(["not a dataframe"])
    except TypeError as e:
        logger.info(f"Correctly caught TypeError: {e}")

