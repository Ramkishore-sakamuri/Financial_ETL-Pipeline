# tests/test_load.py

import unittest
import pandas as pd
import os
import json
import sqlite3
import logging
from sqlalchemy import create_engine, text # For DbLoader verification if needed

# Import classes to be tested
from src.load.file_loader import FileLoader
from src.load.db_loader import DbLoader

# Configure basic logging for tests
# logging.basicConfig(level=logging.DEBUG) # Uncomment for detailed logs

class TestFileLoader(unittest.TestCase):
    """
    Unit tests for the FileLoader class.
    """
    test_output_dir = "test_temp_loader_output"
    sample_df = pd.DataFrame({
        'ColA': [1, 2, 3],
        'ColB': ['apple', 'banana', 'cherry'],
        'ColC': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })

    @classmethod
    def setUpClass(cls):
        """Create a directory for temporary test output files."""
        if not os.path.exists(cls.test_output_dir):
            os.makedirs(cls.test_output_dir)
        cls.loader = FileLoader()

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary test output files and directory."""
        if os.path.exists(cls.test_output_dir):
            for f_name in os.listdir(cls.test_output_dir):
                f_path = os.path.join(cls.test_output_dir, f_name)
                if os.path.isdir(f_path): # Clean up nested dirs if any
                    for nested_f_name in os.listdir(f_path):
                        os.remove(os.path.join(f_path, nested_f_name))
                    os.rmdir(f_path)
                else:
                    os.remove(f_path)
            os.rmdir(cls.test_output_dir)

    def test_save_to_csv_successful(self):
        file_path = os.path.join(self.test_output_dir, "output.csv")
        success = self.loader.save_to_csv(self.sample_df, file_path, index=False)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        # Verify content
        df_read = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(df_read, self.sample_df, check_dtype=False, check_datetimelike_compat=True) # Dtypes might change slightly

    def test_save_to_json_records_successful(self):
        file_path = os.path.join(self.test_output_dir, "output_records.json")
        success = self.loader.save_to_json(self.sample_df, file_path, orient='records', indent=2)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        # Verify content
        with open(file_path, 'r') as f:
            data_read = json.load(f)
        # Convert Timestamps to string for comparison if necessary, as JSON serializes them
        expected_data = self.sample_df.astype({'ColC': str}).to_dict(orient='records')
        # JSON loaded dates will be strings, so compare accordingly or parse them back
        # For simplicity, we'll compare length and a sample value
        self.assertEqual(len(data_read), len(expected_data))
        self.assertEqual(data_read[0]['ColA'], expected_data[0]['ColA'])


    def test_save_to_json_lines_successful(self):
        file_path = os.path.join(self.test_output_dir, "output.jsonl")
        success = self.loader.save_to_json(self.sample_df, file_path, orient='records', lines=True)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        # Verify content (check number of lines)
        line_count = 0
        with open(file_path, 'r') as f:
            for line in f:
                json.loads(line) # Check if each line is valid JSON
                line_count += 1
        self.assertEqual(line_count, len(self.sample_df))

    def test_save_to_parquet_successful(self):
        try:
            import pyarrow # Check if pyarrow is available
            file_path = os.path.join(self.test_output_dir, "output.parquet")
            success = self.loader.save_to_parquet(self.sample_df, file_path, index=False)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(file_path))
            # Verify content
            df_read = pd.read_parquet(file_path)
            pd.testing.assert_frame_equal(df_read, self.sample_df)
        except ImportError:
            self.skipTest("pyarrow not installed, skipping Parquet test.")

    def test_ensure_directory_exists(self):
        nested_dir = os.path.join(self.test_output_dir, "new_nested_dir")
        nested_file_path = os.path.join(nested_dir, "nested_output.csv")
        # Ensure nested_dir does not exist initially for a clean test
        if os.path.exists(nested_dir):
             for f_name in os.listdir(nested_dir): # clean up previous run
                os.remove(os.path.join(nested_dir, f_name))
             os.rmdir(nested_dir)

        success = self.loader.save_to_csv(self.sample_df, nested_file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(nested_file_path))

    def test_save_non_dataframe_fails(self):
        file_path = os.path.join(self.test_output_dir, "fail.csv")
        with self.assertLogs(logger='src.load.file_loader', level='ERROR'):
            success = self.loader.save_to_csv(["not", "a", "dataframe"], file_path)
        self.assertFalse(success)


class TestDbLoader(unittest.TestCase):
    """
    Unit tests for the DbLoader class using a file-based SQLite database.
    """
    db_file_path = os.path.join(os.getcwd(), "test_loader_db.sqlite") # Create in current dir for visibility
    sample_df_db = pd.DataFrame({
        'ID': [101, 102, 103],
        'Name': ['Widget A', 'Gadget B', 'Gizmo C'],
        'Price': [19.99, 120.50, 75.00],
        'LastUpdate': pd.to_datetime(['2023-02-01', '2023-02-15', '2023-03-01'])
    })
    table_name = "products"

    @classmethod
    def setUpClass(cls):
        # Ensure no old DB file exists
        if os.path.exists(cls.db_file_path):
            os.remove(cls.db_file_path)

    @classmethod
    def tearDownClass(cls):
        """Remove the test SQLite database file."""
        if os.path.exists(cls.db_file_path):
            os.remove(cls.db_file_path)

    def setUp(self):
        """Initialize DbLoader for each test."""
        self.db_params = {'database_path': self.db_file_path}
        self.loader = DbLoader(db_type='sqlite', connection_params=self.db_params)
        # Ensure a fresh state for the DB file for some tests by removing it if it exists
        # For tests that depend on prior state (like append), this will be handled in the test itself.

    def tearDown(self):
        """Close loader connection after each test."""
        self.loader.close()
        # For some tests, we might want to delete the db file to ensure isolation
        # if os.path.exists(self.db_file_path):
        #     os.remove(self.db_file_path)


    def _verify_table_content(self, expected_row_count, custom_query=None, params=None):
        """Helper to verify table content."""
        engine = create_engine(f"sqlite:///{self.db_file_path}")
        with engine.connect() as connection:
            if custom_query:
                result_df = pd.read_sql_query(sql=text(custom_query), con=connection, params=params)
                return result_df
            else:
                result_df = pd.read_sql_table(self.table_name, connection)
                self.assertEqual(len(result_df), expected_row_count)
                return result_df
        return pd.DataFrame()


    def test_connect_successful(self):
        self.assertTrue(self.loader.connect())
        self.assertIsNotNone(self.loader.engine)

    def test_load_data_replace(self):
        if os.path.exists(self.db_file_path): os.remove(self.db_file_path) # Fresh DB
        self.assertTrue(self.loader.connect())
        success = self.loader.load_data(self.sample_df_db, self.table_name, if_exists='replace', index=False)
        self.assertTrue(success)
        df_read = self._verify_table_content(expected_row_count=len(self.sample_df_db))
        # Compare ignoring index and types for simplicity, focus on data
        pd.testing.assert_frame_equal(df_read.reset_index(drop=True), self.sample_df_db.reset_index(drop=True), check_dtype=False, check_like=True)


    def test_load_data_append(self):
        if os.path.exists(self.db_file_path): os.remove(self.db_file_path) # Fresh DB for this sequence
        self.assertTrue(self.loader.connect())
        # First load
        self.loader.load_data(self.sample_df_db, self.table_name, if_exists='replace', index=False)
        
        # Append new data
        df_to_append = pd.DataFrame({'ID': [104], 'Name': ['Contraption D'], 'Price': [99.99], 'LastUpdate': pd.to_datetime(['2023-03-05'])})
        success_append = self.loader.load_data(df_to_append, self.table_name, if_exists='append', index=False)
        self.assertTrue(success_append)
        
        expected_total_rows = len(self.sample_df_db) + len(df_to_append)
        self._verify_table_content(expected_row_count=expected_total_rows)

    def test_load_data_fail_if_exists(self):
        if os.path.exists(self.db_file_path): os.remove(self.db_file_path) # Fresh DB
        self.assertTrue(self.loader.connect())
        # First load, table created
        self.loader.load_data(self.sample_df_db, self.table_name, if_exists='replace', index=False)
        
        # Attempt to load again with if_exists='fail'
        with self.assertLogs(logger='src.load.db_loader', level='ERROR'): # Expecting pandas to_sql to raise ValueError
            success_fail = self.loader.load_data(self.sample_df_db, self.table_name, if_exists='fail', index=False)
        self.assertFalse(success_fail) # Should fail as table exists

    def test_load_empty_dataframe(self):
        if os.path.exists(self.db_file_path): os.remove(self.db_file_path) # Fresh DB
        self.assertTrue(self.loader.connect())
        empty_df = pd.DataFrame(columns=self.sample_df_db.columns)
        success = self.loader.load_data(empty_df, self.table_name, if_exists='replace', index=False)
        self.assertTrue(success) # Should be true as it's a no-op
        # Table might be created with 0 rows if 'replace' is used, or not touched if 'append'
        # Let's verify table exists but is empty if 'replace' was used
        if os.path.exists(self.db_file_path): # Check if DB file was created
            self._verify_table_content(expected_row_count=0)


    def test_execute_post_load_sql(self):
        if os.path.exists(self.db_file_path): os.remove(self.db_file_path) # Fresh DB
        self.assertTrue(self.loader.connect())
        self.loader.load_data(self.sample_df_db, self.table_name, if_exists='replace', index=False)

        # For SQLite, loader.connection should be set up by connect()
        if self.loader.db_type == 'sqlite' and not self.loader.connection:
             self.fail("Raw SQLite connection not established by DbLoader.connect()")

        sql_statements = [
            f"UPDATE {self.table_name} SET Price = Price * 1.1 WHERE Name = 'Widget A';",
            f"CREATE INDEX IF NOT EXISTS idx_product_name ON {self.table_name}(Name);"
        ]
        success_post_sql = self.loader.execute_post_load_sql(sql_statements)
        self.assertTrue(success_post_sql)

        # Verify the update and index creation (index check is harder, focus on data)
        updated_df = self._verify_table_content(
            expected_row_count=len(self.sample_df_db),
            custom_query=f"SELECT Price FROM {self.table_name} WHERE Name = 'Widget A'"
        )
        self.assertAlmostEqual(updated_df['Price'].iloc[0], 19.99 * 1.1)

    def test_unsupported_db_type_connect_fail(self):
        loader_fail = DbLoader(db_type="nosuchdb", connection_params={})
        with self.assertLogs(logger='src.load.db_loader', level='ERROR'):
            connected = loader_fail.connect()
        self.assertFalse(connected)
        loader_fail.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
