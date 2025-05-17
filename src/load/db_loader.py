# src/load/db_loader.py

import pandas as pd
import sqlite3
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine # For more robust to_sql with various DBs

# Import other database connectors if needed, e.g.:
# import psycopg2 # For PostgreSQL
# import pyodbc   # For SQL Server or other ODBC sources

# Get a logger for this module
logger = logging.getLogger(__name__)

class DbLoader:
    """
    A class to load pandas DataFrames into various databases.
    """

    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        """
        Initializes the DbLoader.

        Args:
            db_type (str): The type of database (e.g., 'sqlite', 'postgresql', 'sqlserver').
            connection_params (Dict[str, Any]): Parameters required to connect to the database.
                For 'sqlite': {'database_path': 'path/to/your.db'}
                For 'postgresql': {'host': '', 'port': '', 'database': '', 'user': '', 'password': ''}
                For 'sqlserver': {'connection_string': 'DRIVER={...};SERVER=...;DATABASE=...;UID=...;PWD=...'}
                                 or {'server': '', 'database': '', 'username': '', 'password': '', 'driver': ''}
        """
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.engine = None # SQLAlchemy engine for pandas to_sql
        self.connection = None # Raw DBAPI connection if needed for other operations
        logger.info(f"DbLoader initialized for database type: {self.db_type}")

    def _create_engine_str(self) -> Optional[str]:
        """
        Creates a SQLAlchemy connection string based on db_type and connection_params.
        """
        if self.db_type == 'sqlite':
            db_path = self.connection_params.get('database_path')
            if not db_path:
                logger.error("SQLite connection failed: 'database_path' not provided.")
                return None
            return f"sqlite:///{db_path}"
        elif self.db_type == 'postgresql':
            user = self.connection_params.get('user')
            password = self.connection_params.get('password')
            host = self.connection_params.get('host')
            port = self.connection_params.get('port', 5432)
            database = self.connection_params.get('database')
            if not all([user, password, host, database]):
                logger.error("PostgreSQL connection failed: Missing one or more required parameters (user, password, host, database).")
                return None
            # Ensure password is URL encoded if it contains special characters - though SQLAlchemy usually handles this.
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif self.db_type == 'sqlserver':
            # Using pyodbc driver
            conn_str = self.connection_params.get('connection_string')
            if conn_str:
                # SQLAlchemy needs the connection string to be prefixed for pyodbc
                # Example: "mssql+pyodbc:///?odbc_connect={}".format(urllib.parse.quote_plus(conn_str))
                # This requires careful construction of the conn_str or using individual params.
                # For simplicity with individual params:
                server = self.connection_params.get('server')
                database = self.connection_params.get('database')
                user = self.connection_params.get('username')
                password = self.connection_params.get('password')
                driver = self.connection_params.get('driver', 'ODBC Driver 17 for SQL Server').replace(' ', '+') # URL encode driver name spaces

                if not all([server, database]):
                     logger.error("SQL Server connection failed: Missing server or database.")
                     return None
                
                conn_url = f"mssql+pyodbc://{user}:{password}@{server}/{database}?driver={driver}"
                if not user or not password: # For trusted connections
                    conn_url = f"mssql+pyodbc://{server}/{database}?driver={driver}&trusted_connection=yes"
                return conn_url
        else:
            logger.error(f"Unsupported database type for SQLAlchemy engine: {self.db_type}")
            return None


    def connect(self) -> bool:
        """
        Establishes a connection engine for pandas.
        Also sets up a raw DBAPI connection if needed for pre/post SQL.

        Returns:
            bool: True if connection engine was created successfully, False otherwise.
        """
        if self.engine:
            logger.info("SQLAlchemy engine already created.")
            return True

        engine_str = self._create_engine_str()
        if not engine_str:
            return False

        try:
            self.engine = create_engine(engine_str)
            # Test connection (optional, but good practice)
            with self.engine.connect() as test_conn:
                logger.info(f"Successfully created SQLAlchemy engine and tested connection for {self.db_type}.")
            
            # For raw DBAPI connection (e.g., SQLite specific DDL)
            if self.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.connection_params.get('database_path'))
            # Add similar for other DBs if direct DBAPI access is needed beyond pandas.to_sql
            # elif self.db_type == 'postgresql':
            #     self.connection = psycopg2.connect(...)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine for {self.db_type}: {e}", exc_info=True)
            self.engine = None
            self.connection = None
            return False

    def load_data(self,
                  df: pd.DataFrame,
                  table_name: str,
                  if_exists: str = 'fail', # 'fail', 'replace', 'append'
                  index: bool = False,
                  chunksize: Optional[int] = None,
                  dtype: Optional[Dict[str, Any]] = None) -> bool:
        """
        Loads a DataFrame into the specified database table.

        Args:
            df (pd.DataFrame): The DataFrame to load.
            table_name (str): The name of the target table in the database.
            if_exists (str): How to behave if the table already exists.
                             'fail': Raise a ValueError.
                             'replace': Drop the table before inserting new values.
                             'append': Insert new values to the existing table.
            index (bool): Write DataFrame index as a column.
            chunksize (Optional[int]): Number of rows to write at a time. Default None (all rows at once).
                                       Useful for large DataFrames.
            dtype (Optional[Dict[str, Any]]): Dictionary specifying data types for columns.
                                              Example: {'col1': sqlalchemy.types.VARCHAR(length=255)}

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        if not self.engine:
            logger.error("Not connected to the database. Call connect() first to create an engine.")
            if not self.connect(): # Try to connect
                return False
        
        if not isinstance(df, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            return False
        
        if df.empty:
            logger.info(f"Input DataFrame is empty. No data to load into table '{table_name}'.")
            return True # Consider this a success as there's nothing to do.

        logger.info(f"Attempting to load {len(df)} rows into table '{table_name}' with if_exists='{if_exists}'.")

        try:
            df.to_sql(name=table_name,
                      con=self.engine,
                      if_exists=if_exists,
                      index=index,
                      chunksize=chunksize,
                      dtype=dtype)
            logger.info(f"Successfully loaded data into table '{table_name}'.")
            return True
        except ValueError as ve: # Often raised by 'fail' if table exists, or other pandas issues
            logger.error(f"ValueError during data loading for table '{table_name}': {ve}", exc_info=True)
            return False
        except Exception as e: # Catch other potential exceptions (DB errors, etc.)
            logger.error(f"An unexpected error occurred while loading data into table '{table_name}': {e}", exc_info=True)
            return False

    def execute_post_load_sql(self, sql_statements: List[str]) -> bool:
        """
        Executes a list of SQL statements after loading data.
        Requires a raw DBAPI connection to be established.

        Args:
            sql_statements (List[str]): A list of SQL statements to execute.

        Returns:
            bool: True if all statements executed successfully, False otherwise.
        """
        if not self.connection:
            logger.warning("No raw DBAPI connection available for execute_post_load_sql. This method might not work for all DB types without it.")
            # Try to use SQLAlchemy engine for general execution if no raw connection
            if not self.engine:
                logger.error("No engine or connection available.")
                return False
            conn_context = self.engine.connect()
        else:
            conn_context = self.connection # Use the raw connection

        try:
            with conn_context as conn: # conn is either SQLAlchemy Connection or raw DBAPI connection
                if hasattr(conn, 'cursor'): # DBAPI style
                    cursor = conn.cursor()
                    for i, stmt in enumerate(sql_statements):
                        logger.info(f"Executing post-load SQL ({i+1}/{len(sql_statements)}): {stmt[:100]}...")
                        cursor.execute(stmt)
                    if hasattr(conn, 'commit'): # Commit if it's a DBAPI connection that needs it
                        conn.commit()
                    cursor.close()
                else: # SQLAlchemy Core Connection style
                    trans = conn.begin() if hasattr(conn, 'begin') else None # Start transaction if possible
                    for i, stmt in enumerate(sql_statements):
                        logger.info(f"Executing post-load SQL ({i+1}/{len(sql_statements)}): {stmt[:100]}...")
                        conn.execute(stmt) # For SQLAlchemy, text() might be needed: from sqlalchemy import text; conn.execute(text(stmt))
                    if trans:
                        trans.commit()
            logger.info("Successfully executed all post-load SQL statements.")
            return True
        except Exception as e:
            logger.error(f"Error executing post-load SQL: {e}", exc_info=True)
            if hasattr(conn_context, 'rollback') and conn_context != self.engine.connect(): # Don't rollback engine's pooled conn
                 if self.connection and hasattr(self.connection, 'rollback'):
                    self.connection.rollback()
            return False
        finally:
            if conn_context != self.connection and hasattr(conn_context, 'close'): # Close if it was a temporary conn from engine
                conn_context.close()


    def close(self):
        """Closes the database connection engine and raw connection if any."""
        if self.engine:
            try:
                self.engine.dispose()
                logger.info(f"SQLAlchemy engine for {self.db_type} disposed.")
            except Exception as e:
                logger.warning(f"Error disposing SQLAlchemy engine: {e}")
            self.engine = None
        
        if self.connection:
            try:
                self.connection.close()
                logger.info(f"Raw DBAPI connection for {self.db_type} closed.")
            except Exception as e:
                logger.warning(f"Error closing raw DBAPI connection: {e}")
            self.connection = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("-" * 30)
    logger.info("Testing DbLoader with SQLite (in-memory)...")

    # Define connection parameters for an in-memory SQLite database
    db_file = 'test_etl_output.db' # Use a file to inspect after
    sqlite_params = {'database_path': db_file}
    loader = DbLoader(db_type='sqlite', connection_params=sqlite_params)

    # Sample DataFrame to load
    data_to_load = {
        'TransactionIdentifier': ['T001', 'T002', 'T003'],
        'ExecutionTime': pd.to_datetime(['2025-05-01 08:30:15', '2025-05-01 09:15:45', '2025-05-02 11:30:00']),
        'TransactionValue': [55.75, 1500.00, 89.99],
        'Status': ['COMPLETED', 'COMPLETED', 'PENDING']
    }
    df_to_load = pd.DataFrame(data_to_load)
    df_to_load['Status'] = df_to_load['Status'].astype('category')


    if loader.connect():
        logger.info("Connection engine created successfully.")

        # Test 1: Load with if_exists='replace' (should create or replace table)
        logger.info("\n--- Test 1: Loading data with if_exists='replace' ---")
        success_replace = loader.load_data(df_to_load, 'financial_transactions', if_exists='replace', index=False)
        logger.info(f"Test 1 (replace) successful: {success_replace}")

        # Verify by reading data back (using DbExtractor logic conceptually)
        if success_replace:
            try:
                conn_verify = sqlite3.connect(db_file)
                df_verify = pd.read_sql_query("SELECT * FROM financial_transactions", conn_verify)
                logger.info(f"Verification: Loaded data (first load):\n{df_verify}")
                conn_verify.close()
            except Exception as e_verify:
                logger.error(f"Verification failed: {e_verify}")


        # Test 2: Load with if_exists='append'
        logger.info("\n--- Test 2: Loading data with if_exists='append' ---")
        df_append = pd.DataFrame({
            'TransactionIdentifier': ['T004'],
            'ExecutionTime': pd.to_datetime(['2025-05-03 10:00:00']),
            'TransactionValue': [120.00],
            'Status': ['COMPLETED']
        })
        df_append['Status'] = df_append['Status'].astype('category')
        success_append = loader.load_data(df_append, 'financial_transactions', if_exists='append', index=False)
        logger.info(f"Test 2 (append) successful: {success_append}")
        
        if success_append:
            try:
                conn_verify = sqlite3.connect(db_file)
                df_verify_append = pd.read_sql_query("SELECT COUNT(*) as count FROM financial_transactions", conn_verify)
                logger.info(f"Verification: Total rows after append: {df_verify_append['count'].iloc[0]}")
                conn_verify.close()
            except Exception as e_verify:
                logger.error(f"Verification after append failed: {e_verify}")

        # Test 3: Load with if_exists='fail' when table exists
        logger.info("\n--- Test 3: Loading data with if_exists='fail' (table exists) ---")
        success_fail = loader.load_data(df_to_load, 'financial_transactions', if_exists='fail', index=False)
        logger.info(f"Test 3 (fail) successful (expected False): {not success_fail}") # Expecting False

        # Test 4: Post-load SQL
        logger.info("\n--- Test 4: Executing post-load SQL ---")
        post_sql_stmts = [
            "CREATE INDEX IF NOT EXISTS idx_status ON financial_transactions(Status);",
            "UPDATE financial_transactions SET Status = 'PROCESSED' WHERE Status = 'COMPLETED';"
        ]
        # Ensure loader.connection is established for SQLite raw execution
        if loader.db_type == 'sqlite' and not loader.connection: # It should be from connect()
            loader.connection = sqlite3.connect(sqlite_params['database_path'])

        success_post_sql = loader.execute_post_load_sql(post_sql_stmts)
        logger.info(f"Test 4 (post-load SQL) successful: {success_post_sql}")
        if success_post_sql:
            try:
                conn_verify = sqlite3.connect(db_file)
                df_verify_status = pd.read_sql_query("SELECT Status, COUNT(*) as count FROM financial_transactions GROUP BY Status", conn_verify)
                logger.info(f"Verification: Status counts after post-load SQL:\n{df_verify_status}")
                conn_verify.close()
            except Exception as e_verify:
                logger.error(f"Verification after post-load SQL failed: {e_verify}")


        loader.close()
    else:
        logger.error("Failed to create connection engine for testing.")

    # Clean up the test database file
    import os
    if os.path.exists(db_file):
        os.remove(db_file)
        logger.info(f"Cleaned up test database file: {db_file}")

    logger.info("-" * 30)
    logger.info("DbLoader test finished.")

