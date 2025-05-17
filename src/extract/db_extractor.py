# src/extract/db_extractor.py

import sqlite3
import pandas as pd
import logging
from typing import Optional, Dict, Any, List

# Import other database connectors if needed, e.g.:
# import psycopg2 # For PostgreSQL
# import pyodbc   # For SQL Server or other ODBC sources

# Get a logger for this module
logger = logging.getLogger(__name__)

class DbExtractor:
    """
    A class to extract data from various databases.
    Currently supports SQLite, with placeholders for other DBs.
    """

    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        """
        Initializes the DbExtractor.

        Args:
            db_type (str): The type of database (e.g., 'sqlite', 'postgresql', 'sqlserver').
            connection_params (Dict[str, Any]): Parameters required to connect to the database.
                For 'sqlite': {'database_path': 'path/to/your.db'}
                For 'postgresql': {'host': '', 'port': '', 'database': '', 'user': '', 'password': ''}
                For 'sqlserver': {'connection_string': 'DRIVER={ODBC Driver};SERVER=...;DATABASE=...;UID=...;PWD=...'}
                                 or {'server': '', 'database': '', 'username': '', 'password': '', 'driver': ''}
        """
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.connection = None
        self.cursor = None
        logger.info(f"DbExtractor initialized for database type: {self.db_type}")

    def connect(self) -> bool:
        """
        Establishes a connection to the database.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if self.connection:
            logger.info("Connection already established.")
            return True

        try:
            if self.db_type == 'sqlite':
                db_path = self.connection_params.get('database_path')
                if not db_path:
                    logger.error("SQLite connection failed: 'database_path' not provided in connection_params.")
                    return False
                self.connection = sqlite3.connect(db_path)
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to SQLite database: {db_path}")

            # --- Placeholder for PostgreSQL ---
            # elif self.db_type == 'postgresql':
            #     self.connection = psycopg2.connect(
            #         host=self.connection_params.get('host'),
            #         port=self.connection_params.get('port', 5432),
            #         database=self.connection_params.get('database'),
            #         user=self.connection_params.get('user'),
            #         password=self.connection_params.get('password')
            #     )
            #     self.cursor = self.connection.cursor()
            #     logger.info(f"Successfully connected to PostgreSQL database: {self.connection_params.get('database')}")

            # --- Placeholder for SQL Server (using pyodbc) ---
            # elif self.db_type == 'sqlserver':
            #     conn_str = self.connection_params.get('connection_string')
            #     if not conn_str: # Fallback to individual parameters
            #         server = self.connection_params.get('server')
            #         database = self.connection_params.get('database')
            #         username = self.connection_params.get('username')
            #         password = self.connection_params.get('password')
            #         driver = self.connection_params.get('driver', '{SQL Server}') # Default driver
            #         if not all([server, database]):
            #             logger.error("SQL Server connection failed: Insufficient parameters.")
            #             return False
            #         conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};'
            #         if username:
            #             conn_str += f'UID={username};'
            #         if password:
            #             conn_str += f'PWD={password};'
            #
            #     self.connection = pyodbc.connect(conn_str)
            #     self.cursor = self.connection.cursor()
            #     logger.info(f"Successfully connected to SQL Server database.")

            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {self.db_type} database: {e}", exc_info=True)
            self.connection = None
            self.cursor = None
            return False

    def extract_data(self, query: str, params: Optional[Union[Dict, List, Tuple]] = None) -> Optional[pd.DataFrame]:
        """
        Executes a SQL query and fetches data as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[Dict, List, Tuple]]): Parameters to pass to the SQL query (for parameterized queries).

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the query results, or None if an error occurs.
        """
        if not self.connection or not self.cursor:
            logger.error("Not connected to the database. Call connect() first.")
            if not self.connect(): # Try to connect if not already
                 return None


        logger.info(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        if params:
            logger.debug(f"Query parameters: {params}")

        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            if self.db_type == 'sqlite': # For SQLite, fetchall and then construct DataFrame
                rows = self.cursor.fetchall()
                column_names = [description[0] for description in self.cursor.description]
                df = pd.DataFrame(rows, columns=column_names)
            # For other DBs like PostgreSQL with psycopg2, pandas.read_sql can be more direct if not using cursor manually
            # elif self.db_type in ['postgresql', 'sqlserver']:
            #     # If using pandas.read_sql, you might not need to manage the cursor explicitly for reads
            #     # However, this example uses the cursor for consistency.
            #     rows = self.cursor.fetchall()
            #     column_names = [desc[0] for desc in self.cursor.description]
            #     df = pd.DataFrame(rows, columns=column_names)
            else: # Fallback or general approach (might need adjustment based on DB API)
                 # For many DB-API compliant cursors, pandas can read directly from the connection and query
                df = pd.read_sql_query(query, self.connection, params=params)


            logger.info(f"Successfully extracted {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error executing query on {self.db_type} database: {e}", exc_info=True)
            # For some DBs, you might need to rollback on error if it's part of a transaction
            # if self.connection and self.db_type != 'sqlite': # SQLite auto-commits DQL
            #     try:
            #         self.connection.rollback()
            #         logger.info("Transaction rolled back due to query error.")
            #     except Exception as rb_err:
            #         logger.error(f"Error during rollback: {rb_err}")
            return None

    def close(self):
        """Closes the database connection."""
        if self.cursor:
            try:
                self.cursor.close()
                logger.debug("Database cursor closed.")
            except Exception as e:
                logger.warning(f"Error closing cursor: {e}")
            self.cursor = None
        if self.connection:
            try:
                self.connection.close()
                logger.info(f"Database connection to {self.db_type} closed.")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            self.connection = None


if __name__ == '__main__':
    # This is for basic testing of the DbExtractor with an in-memory SQLite DB.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("-" * 30)
    logger.info("Testing DbExtractor with SQLite (in-memory)...")

    # Define connection parameters for an in-memory SQLite database
    sqlite_params = {'database_path': ':memory:'}
    extractor = DbExtractor(db_type='sqlite', connection_params=sqlite_params)

    if extractor.connect():
        logger.info("Connection successful.")

        # Create a sample table and insert data
        create_table_query = """
        CREATE TABLE transactions (
            TransactionID TEXT PRIMARY KEY,
            TransactionDate TEXT,
            Amount REAL,
            Currency TEXT,
            Category TEXT
        );
        """
        insert_data_query = """
        INSERT INTO transactions (TransactionID, TransactionDate, Amount, Currency, Category)
        VALUES (?, ?, ?, ?, ?);
        """
        sample_data = [
            ('TXN001', '2025-05-01', 100.50, 'USD', 'Groceries'),
            ('TXN002', '2025-05-01', 75.20, 'EUR', 'Software'),
            ('TXN003', '2025-05-02', 200.00, 'USD', 'Electronics'),
            ('TXN004', '2025-05-03', 50.00, 'GBP', 'Books')
        ]

        try:
            extractor.cursor.execute(create_table_query)
            logger.info("Table 'transactions' created.")
            extractor.cursor.executemany(insert_data_query, sample_data)
            extractor.connection.commit() # Commit DML changes for SQLite
            logger.info(f"{len(sample_data)} rows inserted into 'transactions'.")
        except Exception as e:
            logger.error(f"Error setting up test table: {e}")

        # Test data extraction
        logger.info("\nExtracting all data from 'transactions' table...")
        all_data_df = extractor.extract_data("SELECT * FROM transactions;")
        if all_data_df is not None:
            print("All Data:")
            print(all_data_df)

        logger.info("\nExtracting USD transactions...")
        usd_transactions_df = extractor.extract_data(
            "SELECT * FROM transactions WHERE Currency = ?;",
            params=('USD',) # Note: params for sqlite3 should be a tuple or list
        )
        if usd_transactions_df is not None:
            print("\nUSD Transactions:")
            print(usd_transactions_df)

        logger.info("\nTesting extraction with an invalid query...")
        invalid_df = extractor.extract_data("SELECT * FROM non_existent_table;")
        if invalid_df is None:
            logger.info("Correctly handled invalid query (returned None, error logged).")

        extractor.close()
    else:
        logger.error("Failed to connect to the in-memory SQLite database for testing.")

    logger.info("-" * 30)
    logger.info("DbExtractor test finished.")
