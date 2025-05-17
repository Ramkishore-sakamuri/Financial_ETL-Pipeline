# src/transform/transaction_transformer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from .data_cleaner import DataCleaner # Assuming data_cleaner.py is in the same directory

# Get a logger for this module
logger = logging.getLogger(__name__)

class TransactionTransformer:
    """
    A class to perform business-specific transformations on financial transaction data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the TransactionTransformer with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be transformed.
                               It's assumed to be relatively clean, but some
                               cleaning steps might be applied here if specific
                               to transformation logic.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy() # Work on a copy
        self.cleaner = DataCleaner(self.df) # Initialize cleaner with the current df
        logger.info(f"TransactionTransformer initialized with DataFrame of shape {self.df.shape}")

    def get_df(self) -> pd.DataFrame:
        """Returns the current state of the transformed DataFrame."""
        self.df = self.cleaner.get_df() # Ensure we get the latest from cleaner
        return self.df

    def create_transaction_timestamp(self,
                                     date_column: str,
                                     time_column: Optional[str] = None,
                                     new_timestamp_column: str = 'TransactionTimestamp',
                                     datetime_format: Optional[str] = None) -> 'TransactionTransformer':
        """
        Combines date and time columns into a single timestamp column.
        If time_column is None, assumes date_column already contains or can be parsed to datetime.

        Args:
            date_column (str): Name of the column containing date information.
            time_column (Optional[str]): Name of the column containing time information.
            new_timestamp_column (str): Name of the new timestamp column to be created.
            datetime_format (Optional[str]): The strftime format if date/time are strings.

        Returns:
            TransactionTransformer: The instance for method chaining.
        """
        logger.info(f"Creating timestamp column '{new_timestamp_column}' from '{date_column}' and '{time_column}'.")
        if date_column not in self.df.columns:
            logger.error(f"Date column '{date_column}' not found.")
            return self
        if time_column and time_column not in self.df.columns:
            logger.error(f"Time column '{time_column}' not found.")
            return self

        try:
            if time_column:
                # Ensure columns are string type before concatenation if they are not already datetime
                self.df[date_column] = self.df[date_column].astype(str)
                self.df[time_column] = self.df[time_column].astype(str)
                datetime_str_series = self.df[date_column] + ' ' + self.df[time_column]
                self.df[new_timestamp_column] = pd.to_datetime(datetime_str_series, format=datetime_format, errors='coerce')
            else: # Only date_column is provided
                self.df[new_timestamp_column] = pd.to_datetime(self.df[date_column], format=datetime_format, errors='coerce')

            if self.df[new_timestamp_column].isnull().any():
                logger.warning(f"Some values in '{new_timestamp_column}' could not be parsed and are NaT.")
            logger.info(f"Successfully created '{new_timestamp_column}'.")
        except Exception as e:
            logger.error(f"Error creating timestamp column '{new_timestamp_column}': {e}", exc_info=True)
        
        self.cleaner.df = self.df # Update cleaner's df
        return self

    def normalize_currency(self,
                           amount_column: str,
                           currency_column: str,
                           target_currency: str = 'USD',
                           exchange_rates: Optional[Dict[str, float]] = None,
                           normalized_amount_column: str = 'NormalizedAmountUSD',
                           is_international_column: Optional[str] = 'IsInternational') -> 'TransactionTransformer':
        """
        Normalizes transaction amounts to a target currency.
        NOTE: This is a conceptual implementation. Real-world usage requires a reliable
              source for up-to-date exchange rates.

        Args:
            amount_column (str): Column with transaction amounts.
            currency_column (str): Column with currency codes (e.g., 'USD', 'EUR').
            target_currency (str): The target currency code.
            exchange_rates (Optional[Dict[str, float]]): Dict of exchange rates to the target_currency.
                                                        Example: {'EUR': 1.1, 'CAD': 0.75} for USD target.
                                                        If None, only creates flag and copies amount if already target.
            normalized_amount_column (str): Name for the new column with normalized amounts.
            is_international_column (Optional[str]): Name for a new boolean column indicating if the transaction
                                                     was not in the target_currency.

        Returns:
            TransactionTransformer: The instance for method chaining.
        """
        logger.info(f"Normalizing currency in '{amount_column}' (original currency in '{currency_column}') to '{target_currency}'.")
        if amount_column not in self.df.columns or currency_column not in self.df.columns:
            logger.error(f"Amount ('{amount_column}') or Currency ('{currency_column}') column not found.")
            return self

        # Ensure amount is numeric
        try:
            self.df[amount_column] = pd.to_numeric(self.df[amount_column], errors='coerce')
            if self.df[amount_column].isnull().any():
                 logger.warning(f"Some values in '{amount_column}' became NaN after numeric conversion during currency normalization.")
        except Exception as e:
            logger.error(f"Could not convert '{amount_column}' to numeric: {e}")
            return self


        # Default to original amount if no rates or currency matches target
        self.df[normalized_amount_column] = self.df[amount_column]

        if exchange_rates:
            for currency_code, rate in exchange_rates.items():
                mask = (self.df[currency_column].str.upper() == currency_code.upper()) & (self.df[amount_column].notna())
                self.df.loc[mask, normalized_amount_column] = self.df.loc[mask, amount_column] * rate
            logger.info(f"Applied exchange rates to create '{normalized_amount_column}'.")
        else:
            logger.warning("No exchange rates provided. Normalized amount will be same as original if currency matches target, otherwise it's the original amount.")

        # Set amount to original if it's already in target currency (covers cases where no rates are given too)
        target_mask = (self.df[currency_column].str.upper() == target_currency.upper()) & (self.df[amount_column].notna())
        self.df.loc[target_mask, normalized_amount_column] = self.df.loc[target_mask, amount_column]


        if is_international_column:
            self.df[is_international_column] = self.df[currency_column].str.upper() != target_currency.upper()
            logger.info(f"Created '{is_international_column}' flag.")

        self.cleaner.df = self.df # Update cleaner's df
        return self

    def standardize_categories(self,
                               category_column: str,
                               mapping: Optional[Dict[str, str]] = None,
                               to_case: Optional[str] = 'upper') -> 'TransactionTransformer':
        """
        Standardizes the transaction category column.

        Args:
            category_column (str): The name of the category column.
            mapping (Optional[Dict[str, str]]): A dictionary to map raw categories to standard ones.
                                                Example: {'grocery_store': 'GROCERIES', 'food_delivery': 'FOOD'}
            to_case (Optional[str]): Convert categories to a specific case ('upper', 'lower', 'title').
                                     Applied after mapping if provided.

        Returns:
            TransactionTransformer: The instance for method chaining.
        """
        logger.info(f"Standardizing category column: '{category_column}'.")
        if category_column not in self.df.columns:
            logger.error(f"Category column '{category_column}' not found.")
            return self

        # Use DataCleaner for basic string cleaning and mapping
        self.cleaner.clean_string_column(category_column, strip_whitespace=True, to_case=None) # Initial strip
        if mapping:
            self.cleaner.map_values(category_column, mapping_dict=mapping, default_value=None) # Unmapped stay as is

        if to_case: # Apply case standardization after mapping
            self.cleaner.clean_string_column(category_column, strip_whitespace=False, to_case=to_case)

        self.df = self.cleaner.get_df() # Refresh self.df from cleaner
        logger.info(f"Finished standardizing '{category_column}'.")
        return self

    def derive_features_from_description(self,
                                         description_column: str,
                                         category_map: Dict[str, List[str]],
                                         new_category_column: str = "DerivedCategory",
                                         default_category: str = "OTHER") -> 'TransactionTransformer':
        """
        Derives a new category based on keywords in the transaction description.

        Args:
            description_column (str): Column containing transaction descriptions.
            category_map (Dict[str, List[str]]): Dictionary where keys are category names
                                                 and values are lists of keywords.
                                                 Example: {'UTILITIES': ['electric', 'gas', 'water'],
                                                           'GROCERIES': ['market', 'supermarket', 'grocery']}
            new_category_column (str): Name of the new column for derived categories.
            default_category (str): Category to assign if no keywords match.

        Returns:
            TransactionTransformer: The instance for method chaining.
        """
        logger.info(f"Deriving features from '{description_column}' into '{new_category_column}'.")
        if description_column not in self.df.columns:
            logger.error(f"Description column '{description_column}' not found.")
            return self

        self.df[new_category_column] = default_category
        # Ensure description is string and lowercase for matching
        descriptions_lower = self.df[description_column].astype(str).str.lower().fillna('')

        for category, keywords in category_map.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                mask = descriptions_lower.str.contains(keyword_lower, na=False, regex=False)
                self.df.loc[mask, new_category_column] = category.upper() # Standardize derived category to upper

        logger.info(f"Successfully derived categories in '{new_category_column}'.")
        self.cleaner.df = self.df # Update cleaner's df
        return self
        
    def select_and_rename_columns(self, column_map: Dict[str, str]) -> 'TransactionTransformer':
        """
        Selects a subset of columns and renames them.
        Columns not in the keys of column_map will be dropped if they are not in the values.

        Args:
            column_map (Dict[str, str]): Dictionary where keys are current column names
                                         and values are new desired column names.
                                         Example: {'TransactionID': 'TxnID', 'Amount': 'TransactionAmount'}
                                         Only columns mentioned as keys (and renamed to values) will be kept.

        Returns:
            TransactionTransformer: The instance for method chaining.
        """
        logger.info(f"Selecting and renaming columns. Target columns: {list(column_map.values())}")
        
        # Check if all old column names exist
        missing_cols = [col for col in column_map.keys() if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"Source columns for renaming not found in DataFrame: {missing_cols}. They will be ignored.")
            # Remove missing columns from the map to avoid errors
            column_map = {k: v for k, v in column_map.items() if k in self.df.columns}

        # Select only the columns that are keys in the column_map and then rename
        self.df = self.df[list(column_map.keys())].rename(columns=column_map)
        
        logger.info(f"Columns selected and renamed. Current columns: {self.df.columns.tolist()}")
        self.cleaner.df = self.df # Update cleaner's df
        return self


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sample data similar to the input CSV structure
    raw_data = {
        'TransactionID': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005'],
        'TransactionDate': ['2025-05-01', '2025-05-01', '2025-05-02', '2025-05-02', '2025-05-03'],
        'TransactionTime': ['08:30:15', '09:15:45', '11:30:00', '14:00:55', '16:50:00'],
        'AccountNumber': ['ACC001', 'ACC002', 'ACC001', 'ACC003', 'ACC002'],
        'TransactionType': ['DEBIT', 'CREDIT', 'DEBIT', 'DEBIT', 'TRANSFER'],
        'Amount': [55.75, 1500.00, '89.99', 22.40, '250.00'], # Mixed type for Amount
        'Currency': ['USD', 'USD', 'EUR', 'USD', 'CAD'],
        'Description': ['Green Valley Groceries', 'Monthly Salary', 'Online Books EU', 'The Coffee House', 'Transfer to Savings'],
        'Category': ['Groceries', 'Salary', 'Shopping', 'Food', 'Transfers'],
        'Status': ['COMPLETED', 'COMPLETED', 'COMPLETED', 'PENDING', 'COMPLETED']
    }
    sample_df = pd.DataFrame(raw_data)
    logger.info("Original DataFrame for Transformer:")
    print(sample_df)
    logger.info(f"Original dtypes:\n{sample_df.dtypes}")

    transformer = TransactionTransformer(sample_df)

    # 1. Create TransactionTimestamp
    logger.info("\n--- Testing Timestamp Creation ---")
    transformer.create_transaction_timestamp(date_column='TransactionDate', time_column='TransactionTime')
    # print(transformer.get_df()['TransactionTimestamp'])

    # 2. Normalize Currency
    logger.info("\n--- Testing Currency Normalization ---")
    exchange_rates_example = {'EUR': 1.08, 'CAD': 0.73} # Example rates to USD
    transformer.normalize_currency(
        amount_column='Amount',
        currency_column='Currency',
        target_currency='USD',
        exchange_rates=exchange_rates_example
    )
    # print(transformer.get_df()[['Amount', 'Currency', 'NormalizedAmountUSD', 'IsInternational']])

    # 3. Standardize Categories (using DataCleaner's methods implicitly)
    logger.info("\n--- Testing Category Standardization ---")
    category_mapping = {
        'Groceries': 'GROCERIES',
        'Salary': 'INCOME',
        'Shopping': 'RETAIL_SHOPPING',
        'Food': 'FOOD_BEVERAGE',
        'Transfers': 'INTERNAL_TRANSFER'
    }
    transformer.standardize_categories(category_column='Category', mapping=category_mapping, to_case='upper')
    # print(transformer.get_df()['Category'])

    # 4. Derive features from description
    logger.info("\n--- Testing Feature Derivation from Description ---")
    description_category_map = {
        'FINANCE': ['salary', 'transfer', 'payment'],
        'SHOPPING': ['groceries', 'books', 'market'],
        'LEISURE': ['coffee', 'restaurant']
    }
    transformer.derive_features_from_description(
        description_column='Description',
        category_map=description_category_map,
        new_category_column='SmartCategory'
    )
    # print(transformer.get_df()[['Description', 'SmartCategory']])

    # 5. Select and Rename Columns for final output
    logger.info("\n--- Testing Column Selection and Renaming ---")
    final_column_map = {
        'TransactionID': 'TxnID',
        'TransactionTimestamp': 'Timestamp',
        'AccountNumber': 'AccountNo',
        'TransactionType': 'Type',
        'Amount': 'OriginalAmount', # Keep original amount
        'Currency': 'OriginalCurrency',
        'NormalizedAmountUSD': 'AmountUSD',
        'IsInternational': 'IsIntlTxn',
        'Description': 'Details',
        'Category': 'StandardCategory', # Renamed from original 'Category'
        'SmartCategory': 'AutoCategory', # Renamed from derived 'SmartCategory'
        'Status': 'TxnStatus'
    }
    transformer.select_and_rename_columns(final_column_map)


    final_transformed_df = transformer.get_df()
    logger.info("\nFinal Transformed DataFrame:")
    print(final_transformed_df)
    logger.info(f"\nFinal Transformed Dtypes:\n{final_transformed_df.dtypes}")
    logger.info(f"\nFinal Transformed Shape: {final_transformed_df.shape}")

