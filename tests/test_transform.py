# tests/test_transform.py

import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import classes to be tested
from src.transform.data_cleaner import DataCleaner
from src.transform.transaction_transformer import TransactionTransformer
from src.transform.schema_mapper import SchemaMapper

# Configure basic logging for tests to see output if needed
# logging.basicConfig(level=logging.DEBUG) # Uncomment for detailed logs

class TestDataCleaner(unittest.TestCase):
    """
    Unit tests for the DataCleaner class.
    """
    def setUp(self):
        """Set up a sample DataFrame for each test."""
        self.raw_data = {
            'ID': [1, 2, 3, 2, 5, 6, None, 8],
            'Name': ['  Alice  ', 'Bob', '  Charlie', 'Bob', 'Eve  ', '  Frank  ', 'Grace', 'Heidi'],
            'Age': [25, 30, None, 30, 28, 35, 40, 22],
            'Salary': [50000.0, 60000.0, 55000.0, 60000.0, np.nan, 80000.0, 75000.0, 62000.0],
            'JoinDate_Str': ['2021-01-10', '2020-05-15', '2021-03-20', '2020-05-15', '2023-11-30', '2019-06-20', '2018-10-10', '2022-07-01'],
            'Category': ['A', 'B', 'A', 'B', '  a  ', ' C', 'B', 'c'],
            'IsActive_Str': ['True', 'False', 'true', 'FALSE', '1', '0', None, 'yes'],
            'Notes': [None, 'Good', 'Excellent', 'Good', 'Okay', 'Super', 'New', 'Pending']
        }
        self.df = pd.DataFrame(self.raw_data)
        self.cleaner = DataCleaner(self.df.copy()) # Operate on a copy

    def test_handle_missing_values_drop_row(self):
        # ID has one NaN, Salary has one NaN
        initial_rows = len(self.cleaner.df)
        self.cleaner.handle_missing_values(strategy='drop_row', subset_for_drop=['ID', 'Salary'])
        df_cleaned = self.cleaner.get_df()
        self.assertEqual(len(df_cleaned), initial_rows - 2) # Drops rows with NaN in ID or Salary
        self.assertFalse(df_cleaned['ID'].isnull().any())
        self.assertFalse(df_cleaned['Salary'].isnull().any())

    def test_handle_missing_values_fill_value(self):
        self.cleaner.handle_missing_values(columns=['Age', 'Notes'], strategy='fill', fill_value=-1)
        df_cleaned = self.cleaner.get_df()
        self.assertEqual(df_cleaned['Age'].iloc[2], -1)
        self.assertEqual(df_cleaned['Notes'].iloc[0], -1)

    def test_handle_missing_values_mean_median_mode(self):
        # Age has one NaN, Salary has one NaN
        mean_age_before_fill = self.cleaner.df['Age'].mean() # Calculate before filling
        median_salary_before_fill = self.cleaner.df['Salary'].median() # Calculate before filling

        self.cleaner.handle_missing_values(columns=['Age'], strategy='mean')
        self.cleaner.handle_missing_values(columns=['Salary'], strategy='median')
        self.cleaner.handle_missing_values(columns=['ID'], strategy='mode') # ID is float due to None

        df_cleaned = self.cleaner.get_df()
        self.assertAlmostEqual(df_cleaned['Age'].iloc[2], mean_age_before_fill)
        self.assertAlmostEqual(df_cleaned['Salary'].iloc[4], median_salary_before_fill)
        self.assertFalse(df_cleaned['ID'].isnull().any()) # Mode fill for ID

    def test_convert_data_types(self):
        column_types = {
            'ID': 'Int64', # Use nullable Int64 for columns with potential NaNs after cleaning
            'Age': 'Int64',
            'Salary': 'float32',
            'IsActive_Str': 'bool' # Test basic bool conversion (cleaner handles simple cases)
        }
        # First, fill NaNs in ID and Age to allow direct Int64 conversion if not already handled
        self.cleaner.df['ID'].fillna(-1, inplace=True)
        self.cleaner.df['Age'].fillna(-1, inplace=True)

        self.cleaner.convert_data_types(column_types)
        df_cleaned = self.cleaner.get_df()
        self.assertTrue(pd.api.types.is_integer_dtype(df_cleaned['ID']))
        self.assertTrue(pd.api.types.is_integer_dtype(df_cleaned['Age']))
        self.assertEqual(df_cleaned['Salary'].dtype, 'float32')
        self.assertEqual(df_cleaned['IsActive_Str'].dtype, 'bool')
        self.assertTrue(df_cleaned['IsActive_Str'].iloc[0]) # 'True' -> True
        self.assertFalse(df_cleaned['IsActive_Str'].iloc[1])# 'False' -> False

    def test_remove_duplicates(self):
        # Row 1 and 3 are duplicates based on ('Name', 'Age', 'Salary') after potential cleaning
        # The setup has ID:2, Name:Bob, Age:30, Salary:60000 duplicated
        initial_rows = len(self.cleaner.df)
        self.cleaner.remove_duplicates(subset=['Name', 'Age', 'Salary'], keep='first')
        df_cleaned = self.cleaner.get_df()
        self.assertEqual(len(df_cleaned), initial_rows - 1)

    def test_clean_string_column(self):
        self.cleaner.clean_string_column('Name', strip_whitespace=True, to_case='title')
        self.cleaner.clean_string_column('Category', strip_whitespace=True, to_case='upper', remove_chars=' ') # Remove internal spaces too
        df_cleaned = self.cleaner.get_df()
        self.assertEqual(df_cleaned['Name'].iloc[0], 'Alice')
        self.assertEqual(df_cleaned['Name'].iloc[4], 'Eve')
        self.assertEqual(df_cleaned['Category'].iloc[4], 'A') # '  a  ' -> 'A'
        self.assertEqual(df_cleaned['Category'].iloc[5], 'C') # ' C' -> 'C'

    def test_map_values(self):
        category_map = {'A': 'Alpha', 'B': 'Beta', 'C': 'Gamma'}
        # Clean Category first to ensure consistent keys for mapping
        self.cleaner.clean_string_column('Category', strip_whitespace=True, to_case='upper')
        self.cleaner.map_values('Category', category_map, default_value='Other')
        df_cleaned = self.cleaner.get_df()
        self.assertEqual(df_cleaned['Category'].iloc[0], 'Alpha') # A -> Alpha
        self.assertEqual(df_cleaned['Category'].iloc[1], 'Beta')  # B -> Beta
        # Assuming '  a  ' became 'A' then 'Alpha'
        self.assertEqual(df_cleaned['Category'].iloc[4], 'Alpha')


    def test_parse_datetime_column(self):
        self.cleaner.parse_datetime_column('JoinDate_Str', datetime_format='%Y-%m-%d')
        df_cleaned = self.cleaner.get_df()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_cleaned['JoinDate_Str']))
        self.assertEqual(df_cleaned['JoinDate_Str'].iloc[0], pd.Timestamp('2021-01-10'))


class TestTransactionTransformer(unittest.TestCase):
    """
    Unit tests for the TransactionTransformer class.
    """
    def setUp(self):
        self.raw_data = {
            'TransactionID': ['T001', 'T002', 'T003'],
            'TxnDate': ['2023-01-15', '2023-01-15', '2023-01-16'],
            'TxnTime': ['10:30:00', '14:45:10', '09:00:05'],
            'Amount': [100.50, 20.00, 75.75],
            'Currency': ['USD', 'EUR', 'USD'],
            'Description': ['Coffee Shop Expense', 'EUR payment for service', 'Book Store Purchase'],
            'RawCategory': ['food', 'services', 'shopping_books']
        }
        self.df = pd.DataFrame(self.raw_data)
        self.transformer = TransactionTransformer(self.df.copy())

    def test_create_transaction_timestamp(self):
        self.transformer.create_transaction_timestamp(date_column='TxnDate', time_column='TxnTime', new_timestamp_column='FullTimestamp')
        df_transformed = self.transformer.get_df()
        self.assertIn('FullTimestamp', df_transformed.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_transformed['FullTimestamp']))
        self.assertEqual(df_transformed['FullTimestamp'].iloc[0], pd.Timestamp('2023-01-15 10:30:00'))

    def test_normalize_currency(self):
        exchange_rates = {'EUR': 1.1} # 1 EUR = 1.1 USD
        self.transformer.normalize_currency(
            amount_column='Amount',
            currency_column='Currency',
            target_currency='USD',
            exchange_rates=exchange_rates,
            normalized_amount_column='AmountUSD',
            is_international_column='IsForeign'
        )
        df_transformed = self.transformer.get_df()
        self.assertIn('AmountUSD', df_transformed.columns)
        self.assertIn('IsForeign', df_transformed.columns)
        self.assertAlmostEqual(df_transformed['AmountUSD'].iloc[0], 100.50) # USD to USD
        self.assertAlmostEqual(df_transformed['AmountUSD'].iloc[1], 20.00 * 1.1) # EUR to USD
        self.assertFalse(df_transformed['IsForeign'].iloc[0])
        self.assertTrue(df_transformed['IsForeign'].iloc[1])

    def test_standardize_categories(self):
        category_map = {'food': 'FOOD_DRINK', 'services': 'SERVICES', 'shopping_books': 'SHOPPING'}
        self.transformer.standardize_categories(category_column='RawCategory', mapping=category_map, to_case='upper')
        df_transformed = self.transformer.get_df()
        self.assertEqual(df_transformed['RawCategory'].iloc[0], 'FOOD_DRINK')
        self.assertEqual(df_transformed['RawCategory'].iloc[2], 'SHOPPING')

    def test_derive_features_from_description(self):
        keyword_map = {'COFFEE_SHOP': ['coffee shop'], 'BOOKS': ['book store']}
        self.transformer.derive_features_from_description(
            description_column='Description',
            category_map=keyword_map,
            new_category_column='DerivedCat',
            default_category='GENERAL_EXPENSE'
        )
        df_transformed = self.transformer.get_df()
        self.assertIn('DerivedCat', df_transformed.columns)
        self.assertEqual(df_transformed['DerivedCat'].iloc[0], 'COFFEE_SHOP')
        self.assertEqual(df_transformed['DerivedCat'].iloc[1], 'GENERAL_EXPENSE') # No keyword match
        self.assertEqual(df_transformed['DerivedCat'].iloc[2], 'BOOKS')

    def test_select_and_rename_columns(self):
        # First, create a timestamp to have a column to select
        self.transformer.create_transaction_timestamp(date_column='TxnDate', time_column='TxnTime', new_timestamp_column='FullTimestamp')
        
        column_selection_map = {
            'TransactionID': 'ID',
            'FullTimestamp': 'DateTime',
            'Amount': 'OriginalAmt',
            'Currency': 'Ccy'
            # Other columns will be dropped
        }
        self.transformer.select_and_rename_columns(column_selection_map)
        df_transformed = self.transformer.get_df()
        
        expected_columns = ['ID', 'DateTime', 'OriginalAmt', 'Ccy']
        self.assertListEqual(list(df_transformed.columns), expected_columns)
        self.assertEqual(df_transformed.shape[1], len(expected_columns))


class TestSchemaMapper(unittest.TestCase):
    """
    Unit tests for the SchemaMapper class.
    """
    def setUp(self):
        self.input_data = {
            'SourceID': ['A1', 'B2', 'C3'],
            'EventDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Value': ['10.5', '20.0', '30.75'],
            'IsActive': ['true', 'FALSE', '1'],
            'OptionalField': [None, 'data', None]
        }
        self.df = pd.DataFrame(self.input_data)

    def test_map_df_successful(self):
        target_schema = [
            {'target_name': 'Identifier', 'source_column': 'SourceID', 'dtype': 'str'},
            {'target_name': 'Timestamp', 'source_column': 'EventDate', 'dtype': 'datetime64[ns]'},
            {'target_name': 'MetricValue', 'source_column': 'Value', 'dtype': 'float64'},
            {'target_name': 'ActiveFlag', 'source_column': 'IsActive', 'dtype': 'bool'},
            {'target_name': 'Notes', 'source_column': 'OptionalField', 'dtype': 'str'}
        ]
        mapper = SchemaMapper(target_schema)
        mapped_df = mapper.map_df(self.df.copy())

        self.assertListEqual(list(mapped_df.columns), ['Identifier', 'Timestamp', 'MetricValue', 'ActiveFlag', 'Notes'])
        self.assertEqual(mapped_df['Identifier'].dtype, 'object') # Pandas uses object for str
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(mapped_df['Timestamp']))
        self.assertEqual(mapped_df['MetricValue'].dtype, 'float64')
        self.assertEqual(mapped_df['ActiveFlag'].dtype, 'bool')
        self.assertTrue(mapped_df['ActiveFlag'].iloc[0]) # 'true' -> True
        self.assertFalse(mapped_df['ActiveFlag'].iloc[1])# 'FALSE' -> False
        self.assertTrue(mapped_df['ActiveFlag'].iloc[2]) # '1' -> True
        self.assertEqual(mapped_df['Notes'].dtype, 'object')

    def test_map_df_missing_source_not_required(self):
        target_schema = [
            {'target_name': 'ID', 'source_column': 'SourceID', 'dtype': 'str'},
            {'target_name': 'MissingData', 'source_column': 'NonExistent', 'dtype': 'str', 'required': False}
        ]
        mapper = SchemaMapper(target_schema)
        mapped_df = mapper.map_df(self.df.copy())
        self.assertIn('MissingData', mapped_df.columns)
        self.assertTrue(mapped_df['MissingData'].isnull().all())
        self.assertEqual(mapped_df['MissingData'].dtype, 'object') # Default for empty string col

    def test_map_df_missing_source_required(self):
        target_schema = [
            {'target_name': 'ID', 'source_column': 'SourceID', 'dtype': 'str'},
            {'target_name': 'CriticalData', 'source_column': 'NonExistentCritical', 'dtype': 'int', 'required': True}
        ]
        mapper = SchemaMapper(target_schema)
        # Expect a log error, column should be created empty
        with self.assertLogs(logger='src.transform.schema_mapper', level='ERROR') as cm:
            mapped_df = mapper.map_df(self.df.copy())
        self.assertIn('CriticalData', mapped_df.columns)
        self.assertTrue(mapped_df['CriticalData'].isnull().all())
        # self.assertTrue(any("CriticalCol_NonExistent' not found" in message for message in cm.output)) # Check specific log
        # Depending on pandas version, dtype of all-NaN int column might be float or object.
        # For 'int' target, if it's all NaN, it will likely be float64 or object.
        # If you used 'Int64', it would be Int64Dtype.
        self.assertTrue(mapped_df['CriticalData'].dtype == 'float64' or mapped_df['CriticalData'].dtype == 'object')


    def test_map_df_type_casting_error(self):
        target_schema = [
            {'target_name': 'ValueAsInt', 'source_column': 'Value', 'dtype': 'int64'} # 'Value' has floats
        ]
        mapper = SchemaMapper(target_schema)
        # Expect a log error, column should remain as is or be object
        with self.assertLogs(logger='src.transform.schema_mapper', level='ERROR'):
            mapped_df = mapper.map_df(self.df.copy())
        self.assertIn('ValueAsInt', mapped_df.columns)
        # Direct astype from float string to int will fail. Pandas might make it object or keep as float.
        # The current SchemaMapper tries astype, if it fails, it's kept as is.
        # '10.5' cannot be directly cast to int64.
        self.assertEqual(mapped_df['ValueAsInt'].dtype, 'object') # Because original 'Value' is object of strings '10.5' etc.

    def test_map_df_empty_input(self):
        target_schema = [
            {'target_name': 'Identifier', 'source_column': 'SourceID', 'dtype': 'str'}
        ]
        mapper = SchemaMapper(target_schema)
        empty_df = pd.DataFrame(columns=self.df.columns) # Same columns, but no rows
        mapped_df = mapper.map_df(empty_df)
        self.assertTrue(mapped_df.empty)
        self.assertListEqual(list(mapped_df.columns), ['Identifier'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
