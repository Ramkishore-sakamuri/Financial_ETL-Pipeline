# tests/test_extract.py

import unittest
import pandas as pd
import os
import json
import sqlite3
import logging
from unittest.mock import patch, MagicMock # For more complex mocking if needed

# Import classes to be tested
# Assuming 'src' is in PYTHONPATH or tests are run from the project root
from src.extract.file_extractor import FileExtractor
from src.extract.api_extractor import ApiExtractor
from src.extract.db_extractor import DbExtractor

# For mocking API calls
import requests_mock

# Configure logging for tests - you might want to suppress or check logs
# For simplicity, we'll let logs go to console if not configured by a main test runner.
# logging.basicConfig(level=logging.DEBUG) # Uncomment to see detailed logs during tests

class TestFileExtractor(unittest.TestCase):
    """
    Unit tests for the FileExtractor class.
    """
    test_dir = "test_temp_data"
    dummy_csv_path = os.path.join(test_dir, "dummy_test.csv")
    dummy_json_path = os.path.join(test_dir, "dummy_test.json")
    empty_csv_path = os.path.join(test_dir, "empty_test.csv")
    malformed_csv_path = os.path.join(test_dir, "malformed_test.csv")
    unsupported_file_path = os.path.join(test_dir, "dummy_test.txt")

    @classmethod
    def setUpClass(cls):
        """Create a directory for temporary test files."""
        if not os.path.exists(cls.test_dir):
            os.makedirs(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary test files and directory."""
        if os.path.exists(cls.dummy_csv_path):
            os.remove(cls.dummy_csv_path)
        if os.path.exists(cls.dummy_json_path):
            os.remove(cls.dummy_json_path)
        if os.path.exists(cls.empty_csv_path):
            os.remove(cls.empty_csv_path)
        if os.path.exists(cls.malformed_csv_path):
            os.remove(cls.malformed_csv_path)
        if os.path.exists(cls.unsupported_file_path):
            os.remove(cls.unsupported_file_path)
        if os.path.exists(cls.test_dir):
            os.rmdir(cls.test_dir)

    def setUp(self):
        """Create dummy files for each test."""
        # Dummy CSV
        with open(self.dummy_csv_path, 'w') as f:
            f.write("ID,Name,Value\n1,Alice,100\n2,Bob,150\n")
        # Dummy JSON
        json_data = [{"ID": 1, "Name": "Charlie", "Value": 200}, {"ID": 2, "Name": "David", "Value": 250}]
        with open(self.dummy_json_path, 'w') as f:
            json.dump(json_data, f)
        # Empty CSV
        with open(self.empty_csv_path, 'w') as f:
            f.write("ID,Name,Value\n") # Header only
        # Malformed CSV
        with open(self.malformed_csv_path, 'w') as f:
            f.write("ID,Name,Value\n1,Eve\"unclosed,300\n")
        # Unsupported file
        with open(self.unsupported_file_path, 'w') as f:
            f.write("This is a text file.")

    def test_csv_extraction(self):
        extractor = FileExtractor(self.dummy_csv_path)
        data = extractor.extract_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        self.assertListEqual(list(data.columns), ["ID", "Name", "Value"])
        self.assertEqual(data.iloc[0]['Name'], "Alice")

    def test_csv_extraction_chunked(self):
        extractor = FileExtractor(self.dummy_csv_path)
        chunks = extractor.extract_data(chunk_size=1)
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 2)
        self.assertIsInstance(chunks[0], pd.DataFrame)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(chunks[0].iloc[0]['Name'], "Alice")

    def test_json_extraction(self):
        extractor = FileExtractor(self.dummy_json_path)
        data = extractor.extract_data() # Returns list of dicts
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data[0], dict)
        self.assertEqual(data[0]['Name'], "Charlie")

    def test_empty_csv_extraction(self):
        extractor = FileExtractor(self.empty_csv_path)
        data = extractor.extract_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(data.empty) # Should be empty but with columns

    def test_malformed_csv_extraction(self):
        # Suppress expected error logs during this specific test for cleaner output
        logging.disable(logging.ERROR)
        extractor = FileExtractor(self.malformed_csv_path)
        data = extractor.extract_data()
        logging.disable(logging.NOTSET) # Re-enable logging
        self.assertIsNone(data) # Expecting None due to ParserError

    def test_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            FileExtractor("non_existent_file.csv")

    def test_unsupported_file_type(self):
        extractor = FileExtractor(self.unsupported_file_path)
        data = extractor.extract_data()
        self.assertIsNone(data)


@requests_mock.Mocker()
class TestApiExtractor(unittest.TestCase):
    """
    Unit tests for the ApiExtractor class.
    Uses requests_mock to simulate API responses.
    """
    base_url = "http://fakeapi.com/api/v1"

    def test_get_single_record_success(self, m):
        mock_response_data = {"id": 1, "name": "Test Item", "value": 123}
        m.get(f"{self.base_url}/items/1", json=mock_response_data, status_code=200)

        extractor = ApiExtractor(base_url=self.base_url)
        data = extractor.get_single_record(endpoint="items/1")
        extractor.close_session()

        self.assertIsNotNone(data)
        self.assertEqual(data["name"], "Test Item")

    def test_get_single_record_not_found(self, m):
        m.get(f"{self.base_url}/items/2", status_code=404, json={"error": "Not Found"})
        
        extractor = ApiExtractor(base_url=self.base_url)
        data = extractor.get_single_record(endpoint="items/2")
        extractor.close_session()
        
        self.assertIsNone(data) # Current implementation returns None on non-200

    def test_fetch_all_data_simple_list(self, m):
        mock_response_data = [{"id": 1, "data": "A"}, {"id": 2, "data": "B"}]
        m.get(f"{self.base_url}/data", json=mock_response_data, status_code=200)

        extractor = ApiExtractor(base_url=self.base_url)
        all_data = extractor.fetch_all_data(endpoint="data")
        extractor.close_session()

        self.assertEqual(len(all_data), 2)
        self.assertEqual(all_data[0]["data"], "A")

    def test_fetch_all_data_with_results_key(self, m):
        mock_response_data = {"results": [{"id": 1}, {"id": 2}], "count": 2}
        m.get(f"{self.base_url}/items", json=mock_response_data, status_code=200)

        extractor = ApiExtractor(base_url=self.base_url)
        all_data = extractor.fetch_all_data(endpoint="items", results_key="results")
        extractor.close_session()

        self.assertEqual(len(all_data), 2)
        self.assertEqual(all_data[0]["id"], 1)

    def test_fetch_all_data_paginated_by_next_url(self, m):
        page1_data = {"results": [{"id": 1}], "next": f"{self.base_url}/items?page=2"}
        page2_data = {"results": [{"id": 2}], "next": None}
        
        m.get(f"{self.base_url}/items", json=page1_data, status_code=200)
        m.get(f"{self.base_url}/items?page=2", json=page2_data, status_code=200)

        extractor = ApiExtractor(base_url=self.base_url)
        all_data = extractor.fetch_all_data(endpoint="items", results_key="results", next_page_url_key="next")
        extractor.close_session()

        self.assertEqual(len(all_data), 2)
        self.assertEqual(all_data[1]["id"], 2)

    def test_fetch_all_data_paginated_by_page_param(self, m):
        m.get(f"{self.base_url}/things?_page=1&_limit=1", json=[{"id": "t1"}], status_code=200)
        m.get(f"{self.base_url}/things?_page=2&_limit=1", json=[{"id": "t2"}], status_code=200)
        m.get(f"{self.base_url}/things?_page=3&_limit=1", json=[], status_code=200) # Empty page to stop

        extractor = ApiExtractor(base_url=self.base_url)
        all_data = extractor.fetch_all_data(
            endpoint="things",
            page_param_name="_page",
            limit_param_name="_limit",
            limit_per_page=1
        )
        extractor.close_session()
        self.assertEqual(len(all_data), 2)
        self.assertEqual(all_data[1]["id"], "t2")

    def test_api_key_auth_header(self, m):
        api_key = "testkey123"
        header_name = "X-API-KEY"
        m.get(f"{self.base_url}/securedata", request_headers={header_name: api_key}, json={"data": "secret"}, status_code=200)

        extractor = ApiExtractor(base_url=self.base_url, api_key=api_key, api_key_header=header_name)
        data = extractor.get_single_record(endpoint="securedata")
        extractor.close_session()

        self.assertIsNotNone(data)
        self.assertEqual(data["data"], "secret")
        self.assertTrue(m.called)
        self.assertEqual(m.last_request.headers[header_name], api_key)
        
    def test_bearer_token_auth_header(self,m):
        token = "bearertokenabc"
        m.get(f"{self.base_url}/authed", request_headers={'Authorization': f'Bearer {token}'}, json={"status": "ok"}, status_code=200)
        
        extractor = ApiExtractor(base_url=self.base_url, auth_token=token)
        data = extractor.get_single_record(endpoint="authed")
        extractor.close_session()
        
        self.assertIsNotNone(data)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(m.last_request.headers['Authorization'], f'Bearer {token}')

    def test_retry_mechanism_on_server_error(self, m):
        # Fail twice, then succeed
        m.get(f"{self.base_url}/retry_endpoint", [
            {'status_code': 500, 'json': {'error': 'Server Down'}},
            {'status_code': 503, 'json': {'error': 'Service Unavailable'}},
            {'status_code': 200, 'json': {'data': 'finally success'}}
        ])

        # Reduce retry delay for faster test, max_retries=2 means 3 total attempts
        extractor = ApiExtractor(base_url=self.base_url, max_retries=2, retry_delay=0.1)
        data = extractor.get_single_record(endpoint="retry_endpoint")
        extractor.close_session()

        self.assertIsNotNone(data)
        self.assertEqual(data['data'], 'finally success')
        self.assertEqual(m.call_count, 3)


class TestDbExtractor(unittest.TestCase):
    """
    Unit tests for the DbExtractor class using an in-memory SQLite database.
    """
    db_name = ":memory:" # Use in-memory SQLite for tests

    def setUp(self):
        """Set up an in-memory SQLite database and populate it."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_transactions (
                id INTEGER PRIMARY KEY,
                description TEXT,
                amount REAL
            )
        """)
        self.cursor.execute("INSERT INTO test_transactions (description, amount) VALUES (?, ?)", ("Sale 1", 100.50))
        self.cursor.execute("INSERT INTO test_transactions (description, amount) VALUES (?, ?)", ("Sale 2", 75.25))
        self.conn.commit()

        # DbExtractor parameters
        self.db_params = {'database_path': self.db_name}
        self.extractor = DbExtractor(db_type='sqlite', connection_params=self.db_params)

    def tearDown(self):
        """Close the database connection."""
        if self.extractor:
            self.extractor.close() # Close extractor's connection
        if self.conn:
            self.conn.close() # Close test setup connection

    def test_connect_success(self):
        connected = self.extractor.connect()
        self.assertTrue(connected)
        self.assertIsNotNone(self.extractor.connection)
        self.assertIsNotNone(self.extractor.cursor)

    def test_extract_all_data(self):
        self.extractor.connect()
        df = self.extractor.extract_data("SELECT * FROM test_transactions ORDER BY id;")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['description'], "Sale 1")
        self.assertEqual(df.iloc[1]['amount'], 75.25)

    def test_extract_data_with_parameters(self):
        self.extractor.connect()
        query = "SELECT * FROM test_transactions WHERE amount > ?;"
        df = self.extractor.extract_data(query, params=(100.0,)) # Tuple for sqlite params
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['description'], "Sale 1")

    def test_extract_data_no_results(self):
        self.extractor.connect()
        df = self.extractor.extract_data("SELECT * FROM test_transactions WHERE amount < 0;")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_extract_data_invalid_query(self):
        # Suppress expected error logs
        logging.disable(logging.ERROR)
        self.extractor.connect()
        df = self.extractor.extract_data("SELECT * FROM non_existent_table;")
        logging.disable(logging.NOTSET) # Re-enable
        self.assertIsNone(df) # Expecting None due to SQL error

    def test_connect_fail_wrong_db_type(self):
        # Suppress expected error logs
        logging.disable(logging.ERROR)
        extractor_fail = DbExtractor(db_type='nonexistentdb', connection_params={})
        connected = extractor_fail.connect()
        logging.disable(logging.NOTSET) # Re-enable
        self.assertFalse(connected)
        extractor_fail.close()

    def test_extract_data_without_connecting(self):
        # The DbExtractor's extract_data now tries to connect if not connected
        # So this test checks if it successfully connects and extracts
        df = self.extractor.extract_data("SELECT * FROM test_transactions WHERE id = 1;")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
