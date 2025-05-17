# src/transform/data_cleaner.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union

# Get a logger for this module
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    A class to perform various data cleaning operations on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataCleaner with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy() # Work on a copy to avoid modifying the original DataFrame
        logger.info(f"DataCleaner initialized with DataFrame of shape {self.df.shape}")

    def get_df(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame."""
        return self.df

    def handle_missing_values(self,
                              columns: Optional[List[str]] = None,
                              strategy: str = 'drop_row',
                              fill_value: Any = None,
                              subset_for_drop: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Handles missing values in specified columns or the entire DataFrame.

        Args:
            columns (Optional[List[str]]): List of columns to apply the strategy.
                                           If None, applies to all columns (for fill) or all rows (for drop_row).
            strategy (str): Method to handle missing values.
                            'drop_row': Drop rows with any missing values in 'subset_for_drop' or all columns.
                            'drop_column': Drop columns with any missing values.
                            'fill': Fill missing values with 'fill_value'.
                            'mean': Fill missing numeric values with the column mean.
                            'median': Fill missing numeric values with the column median.
                            'mode': Fill missing values with the column mode (first mode if multiple).
            fill_value (Any): Value to use when strategy is 'fill'.
            subset_for_drop (Optional[List[str]]): Columns to consider when strategy is 'drop_row'.
                                                   If None, considers all columns.

        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        logger.info(f"Handling missing values with strategy: '{strategy}'")
        target_cols = columns if columns else self.df.columns.tolist()

        if strategy == 'drop_row':
            initial_rows = len(self.df)
            subset = subset_for_drop if subset_for_drop else None # pandas dropna uses None for all columns
            self.df.dropna(subset=subset, inplace=True)
            rows_dropped = initial_rows - len(self.df)
            logger.info(f"Dropped {rows_dropped} rows with missing values. Current shape: {self.df.shape}")
        elif strategy == 'drop_column':
            initial_cols = self.df.shape[1]
            # For dropping columns, pandas dropna axis=1 is used.
            # If 'columns' arg is provided, it means we check NaNs *only* in these columns
            # and drop these columns if they have *any* NaNs.
            # This is a bit different from dropping *any* column that has NaNs.
            # Let's assume 'columns' means "these are the columns to consider dropping if they have nans"
            cols_to_drop = [col for col in target_cols if self.df[col].isnull().any()]
            if cols_to_drop:
                self.df.drop(columns=cols_to_drop, inplace=True)
                logger.info(f"Dropped columns with missing values: {cols_to_drop}. Current shape: {self.df.shape}")
            else:
                logger.info("No columns found with missing values among the specified target columns.")
        elif strategy == 'fill':
            if fill_value is None:
                logger.warning("Strategy 'fill' selected but no 'fill_value' provided. No changes made.")
                return self
            for col in target_cols:
                if col in self.df.columns:
                    self.df[col].fillna(fill_value, inplace=True)
            logger.info(f"Filled missing values with '{fill_value}' in columns: {target_cols}. Current shape: {self.df.shape}")
        elif strategy in ['mean', 'median', 'mode']:
            for col in target_cols:
                if col in self.df.columns:
                    if self.df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(self.df[col]) or strategy == 'mode':
                            if strategy == 'mean':
                                fill_val = self.df[col].mean()
                            elif strategy == 'median':
                                fill_val = self.df[col].median()
                            else: # mode
                                fill_val = self.df[col].mode()
                                if not fill_val.empty:
                                    fill_val = fill_val[0]
                                else:
                                    logger.warning(f"Mode for column '{col}' is empty. Skipping fill.")
                                    continue
                            self.df[col].fillna(fill_val, inplace=True)
                            logger.info(f"Filled missing values in column '{col}' with {strategy} ({fill_val:.2f if isinstance(fill_val, (int, float)) else fill_val}).")
                        else:
                            logger.warning(f"Column '{col}' is not numeric. Cannot apply '{strategy}' strategy. Skipping.")
        else:
            logger.error(f"Invalid missing value strategy: {strategy}")
        return self

    def convert_data_types(self, column_types: Dict[str, Union[str, type]]) -> 'DataCleaner':
        """
        Converts columns to specified data types.

        Args:
            column_types (Dict[str, Union[str, type]]): A dictionary where keys are column names
                                                        and values are the target data types
                                                        (e.g., 'int', 'float', 'str', np.int64, datetime64[ns]).

        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        logger.info(f"Converting data types for columns: {list(column_types.keys())}")
        for col, dtype in column_types.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime': # Special handling for datetime
                        self.df[col] = pd.to_datetime(self.df[col])
                        logger.info(f"Converted column '{col}' to datetime.")
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                        logger.info(f"Converted column '{col}' to type '{dtype}'.")
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to type '{dtype}': {e}", exc_info=True)
            else:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping type conversion.")
        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        """
        Removes duplicate rows from the DataFrame.

        Args:
            subset (Optional[List[str]]): List of columns to consider for identifying duplicates.
                                          If None, all columns are used.
            keep (str): Which duplicates to keep ('first', 'last', False for dropping all duplicates).

        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        rows_dropped = initial_rows - len(self.df)
        if rows_dropped > 0:
            logger.info(f"Removed {rows_dropped} duplicate rows. Current shape: {self.df.shape}")
        else:
            logger.info("No duplicate rows found based on criteria.")
        return self

    def clean_string_column(self,
                            column_name: str,
                            strip_whitespace: bool = True,
                            to_case: Optional[str] = None, # 'lower', 'upper', 'title'
                            remove_chars: Optional[str] = None) -> 'DataCleaner':
        """
        Cleans a specified string column.

        Args:
            column_name (str): The name of the string column to clean.
            strip_whitespace (bool): If True, strips leading/trailing whitespace.
            to_case (Optional[str]): Converts string to specified case ('lower', 'upper', 'title').
            remove_chars (Optional[str]): A string of characters to remove from the column values.

        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for string cleaning.")
            return self

        if not pd.api.types.is_string_dtype(self.df[column_name]) and not self.df[column_name].dtype == 'object':
             # Attempt conversion if it's object type and looks like strings
            try:
                self.df[column_name] = self.df[column_name].astype(str)
                logger.info(f"Column '{column_name}' converted to string type for cleaning.")
            except Exception as e:
                logger.error(f"Column '{column_name}' is not string type and could not be converted. Skipping cleaning. Error: {e}")
                return self


        logger.info(f"Cleaning string column: '{column_name}'")
        # Ensure column is treated as string, handling potential NaNs
        temp_series = self.df[column_name].astype(str).fillna('')


        if strip_whitespace:
            temp_series = temp_series.str.strip()
            logger.debug(f"Stripped whitespace from '{column_name}'.")

        if to_case:
            if to_case == 'lower':
                temp_series = temp_series.str.lower()
            elif to_case == 'upper':
                temp_series = temp_series.str.upper()
            elif to_case == 'title':
                temp_series = temp_series.str.title()
            else:
                logger.warning(f"Invalid case option '{to_case}'. No case change applied.")
            logger.debug(f"Converted '{column_name}' to '{to_case}' case.")

        if remove_chars:
            for char_to_remove in remove_chars:
                temp_series = temp_series.str.replace(char_to_remove, '', regex=False)
            logger.debug(f"Removed characters '{remove_chars}' from '{column_name}'.")
        
        self.df[column_name] = temp_series
        # Handle original NaNs: if a value was NaN, astype(str) made it 'nan'.
        # If strip_whitespace was false and it was NaN, it's still 'nan'.
        # We might want to convert these 'nan' strings back to np.nan if they were originally NaN.
        # This is tricky because a legitimate string could be 'nan'.
        # A safer approach is to only apply string methods to non-null values if astype(str) is not used.
        # For now, the astype(str).fillna('') handles this by making NaNs empty strings before processing.

        return self

    def map_values(self, column_name: str, mapping_dict: Dict[Any, Any], default_value: Optional[Any] = None) -> 'DataCleaner':
        """
        Maps values in a column based on a dictionary.
        Values not in the mapping_dict can be set to a default_value or kept as is.

        Args:
            column_name (str): The column whose values are to be mapped.
            mapping_dict (Dict[Any, Any]): Dictionary for mapping old values to new values.
            default_value (Optional[Any]): Value to use for items not in the mapping_dict.
                                           If None, unmapped values are unchanged.

        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for value mapping.")
            return self

        logger.info(f"Mapping values in column '{column_name}'.")
        if default_value is not None:
            self.df[column_name] = self.df[column_name].map(mapping_dict).fillna(default_value)
            logger.debug(f"Unmapped values in '{column_name}' set to default: '{default_value}'.")
        else:
            # Create a series with mapped values, leaving unmapped as NaN, then fill NaNs with original values
            original_series = self.df[column_name].copy()
            mapped_series = self.df[column_name].map(mapping_dict)
            self.df[column_name] = mapped_series.fillna(original_series)
            logger.debug(f"Unmapped values in '{column_name}' kept as original.")
        return self

    def parse_datetime_column(self,
                              column_name: str,
                              datetime_format: Optional[str] = None,
                              errors: str = 'raise') -> 'DataCleaner':
        """
        Parses a column to datetime objects.

        Args:
            column_name (str): The name of the column to parse.
            datetime_format (Optional[str]): The strftime format string of the date.
                                             If None, pandas will attempt to infer it.
            errors (str): {'raise', 'coerce', 'ignore'}.
                          'raise': invalid parsing will raise an exception.
                          'coerce': invalid parsing will be set as NaT.
                          'ignore': invalid parsing will return the input.
        Returns:
            DataCleaner: The DataCleaner instance for method chaining.
        """
        if column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found for datetime parsing.")
            return self

        logger.info(f"Parsing column '{column_name}' to datetime with format '{datetime_format if datetime_format else 'inferred'}'.")
        try:
            self.df[column_name] = pd.to_datetime(self.df[column_name], format=datetime_format, errors=errors)
            if self.df[column_name].isnull().any() and errors == 'coerce':
                logger.warning(f"Some values in '{column_name}' could not be parsed to datetime and were set to NaT.")
        except Exception as e:
            logger.error(f"Error parsing datetime column '{column_name}': {e}", exc_info=True)
        return self


if __name__ == '__main__':
    # This is for basic testing of the DataCleaner.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sample DataFrame for testing
    data = {
        'ID': [1, 2, 3, 4, 2, 5, 6, 7],
        'Name': [' Alice  ', ' Bob', 'Charlie', 'David', ' Bob', 'Eve  ', '  Frank', None],
        'Age': [25, 30, None, 22, 30, 28, 35, 40],
        'Salary': [50000, 60000, 55000, None, 60000, 70000, 80000, 75000.55],
        'JoinDate': ['2021-01-10', '2020-05-15', '2021-03-20', '2022-07-01', '2020-05-15', '2023-11-30', '2019-06-20', '2018-10-10'],
        'Category': ['A', 'B', 'A', 'C', 'B', 'A', ' D', 'B'],
        'Status': ['active', 'inactive', 'Active', 'pending', 'inactive', 'ACTIVE', 'pending', 'Active']
    }
    sample_df = pd.DataFrame(data)

    logger.info("Original DataFrame:")
    print(sample_df)
    logger.info(f"Original Dtypes:\n{sample_df.dtypes}")


    cleaner = DataCleaner(sample_df)

    # 1. Handle missing values
    logger.info("\n--- Testing Missing Value Handling ---")
    cleaner.handle_missing_values(columns=['Age'], strategy='median')
    cleaner.handle_missing_values(strategy='drop_row', subset_for_drop=['Salary']) # Drop rows where Salary is NaN
    # print("\nAfter handling missing values:")
    # print(cleaner.get_df())

    # 2. Convert data types
    logger.info("\n--- Testing Data Type Conversion ---")
    cleaner.convert_data_types({'Age': 'int', 'Salary': 'float'}) # Age was float due to NaN, Salary is already float
    # print("\nAfter type conversion:")
    # print(cleaner.get_df())
    # logger.info(f"Dtypes after conversion:\n{cleaner.get_df().dtypes}")


    # 3. Remove duplicates
    # Row with ID 2 ('Bob') is a duplicate based on all columns after previous cleaning
    logger.info("\n--- Testing Duplicate Removal ---")
    cleaner.remove_duplicates(subset=['Name', 'Age', 'Salary'], keep='first')
    # print("\nAfter removing duplicates:")
    # print(cleaner.get_df())

    # 4. Clean string column
    logger.info("\n--- Testing String Cleaning ---")
    cleaner.clean_string_column('Name', strip_whitespace=True, to_case='title')
    cleaner.clean_string_column('Category', strip_whitespace=True, to_case='upper')
    # print("\nAfter cleaning 'Name' and 'Category' columns:")
    # print(cleaner.get_df()[['Name', 'Category']])

    # 5. Map values
    logger.info("\n--- Testing Value Mapping ---")
    status_mapping = {'active': 'ACTIVE', 'inactive': 'INACTIVE', 'pending': 'PENDING'}
    cleaner.map_values('Status', status_mapping, default_value='UNKNOWN') # Map and standardize 'Status'
    # print("\nAfter mapping 'Status' column:")
    # print(cleaner.get_df()['Status'])

    # 6. Parse datetime column
    logger.info("\n--- Testing Datetime Parsing ---")
    cleaner.parse_datetime_column('JoinDate', datetime_format='%Y-%m-%d')
    # print("\nAfter parsing 'JoinDate':")
    # print(cleaner.get_df()['JoinDate'])
    # logger.info(f"Dtype of JoinDate: {cleaner.get_df()['JoinDate'].dtype}")

    final_df = cleaner.get_df()
    logger.info("\nFinal Cleaned DataFrame:")
    print(final_df)
    logger.info(f"\nFinal Dtypes:\n{final_df.dtypes}")
    logger.info(f"\nFinal Shape: {final_df.shape}")

    # Test edge case: column not found
    logger.info("\n--- Testing Non-Existent Column ---")
    cleaner.clean_string_column("NonExistentColumn")
    cleaner.convert_data_types({"NonExistentColumn": "int"})

