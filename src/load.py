import pandas as pd
from sqlalchemy import create_engine, text
from .utils.db_utils import get_db_engine # Assuming you'll create this
import logging
import time

logger = logging.getLogger(__name__)

def load_data_to_db(df_transformed, config):
    """
    Loads the transformed DataFrame into the target database.
    - Uses SQLAlchemy for database interaction.
    - Supports 'append' and 'replace' load methods.
    - Uses pandas `to_sql` with `method='multi'` for efficient batch inserts.
      This can significantly improve load times compared to row-by-row inserts.

    Args:
        df_transformed (pandas.DataFrame): The transformed data.
        config (dict): The ETL configuration dictionary.

    Returns:
        bool: True if load was successful, False otherwise.
    """
    if df_transformed is None or df_transformed.empty:
        logger.warning("No data to load. Skipping load process.")
        return True # Or False, depending on desired behavior for empty dataframes

    db_config = config['target_database']
    table_name = db_config['target_table']
    load_method = db_config.get('load_method', 'append')
    batch_size = db_config.get('batch_size', 1000) # Chunk size for to_sql

    logger.info(f"Starting data load to table '{table_name}' using method '{load_method}'. {len(df_transformed)} rows.")
    
    try:
        engine = get_db_engine(db_config) # Get SQLAlchemy engine from db_utils
        with engine.connect() as connection:
            start_time = time.time()
            
            # Performance Tip: For 'replace' on very large tables,
            # it might be faster to TRUNCATE and then APPEND, or drop and recreate.
            # This depends on the database system and table structure (indexes, constraints).
            if load_method == 'replace':
                logger.info(f"Replacing data in table {table_name}. This might involve dropping and recreating.")
                # Option 1: Truncate (if supported and desired)
                # try:
                #     connection.execute(text(f"TRUNCATE TABLE {table_name}")) # Use with caution!
                #     connection.commit()
                #     logger.info(f"Table {table_name} truncated.")
                # except Exception as e:
                #     logger.warning(f"Could not truncate table {table_name}: {e}. Will use if_exists='replace'.")
                #     pass # Fallback to pandas default 'replace' behavior

            # Using method='multi' for to_sql is generally much faster than the default
            # as it sends multiple rows in a single INSERT statement.
            # `chunksize` further helps manage memory for very large DataFrames.
            df_transformed.to_sql(
                name=table_name,
                con=engine, # Pass engine directly, to_sql handles connection
                if_exists=load_method, # 'append' or 'replace'
                index=False,
                chunksize=batch_size,
                method='multi' # Key for performance
            )
            
            # For PostgreSQL, using psycopg2.extras.execute_values can be even faster for appends.
            # This would require a more custom implementation than just df.to_sql.
            # Example (conceptual, would be in db_utils or here):
            # if db_config['db_type'] == 'postgresql' and load_method == 'append':
            #     from psycopg2.extras import execute_values
            #     # ... (get raw psycopg2 connection) ...
            #     # ... (convert df to list of tuples) ...
            #     # execute_values(cursor, "INSERT INTO ... VALUES %s", list_of_tuples)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Data loaded successfully to '{table_name}' in {duration:.2f} seconds.")
            
            # --- How this contributes to "40% faster load times" ---
            # 1. `method='multi'` in `to_sql` sends records in batches, reducing network overhead
            #    and database commit frequency compared to single-row inserts.
            # 2. `chunksize` helps manage memory on the client-side for very large dataframes,
            #    preventing crashes and allowing smoother streaming to the DB.
            # 3. Compared to a naive loop inserting row-by-row, this approach is orders of magnitude faster.
            #    If the "old" pipeline was row-by-row, a 40% (or much higher) improvement is very likely.
            # 4. For 'replace', if the old method involved deleting rows one by one or inefficiently,
            #    `if_exists='replace'` or a TRUNCATE strategy is much faster.

            return True

    except Exception as e:
        logger.error(f"Error during data load to '{table_name}': {e}", exc_info=True)
        return False

if __name__ == '__main__':
    import yaml
    from .utils.file_utils import get_project_root
    import os
    # Example Usage (for testing this module directly)
    project_r = get_project_root()
    with open(os.path.join(project_r, 'config/etl_config.yml'), 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=config.get('logging', {}).get('level', 'INFO'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a dummy transformed DataFrame for testing
    # In a real scenario, this would come from the transform step.
    dummy_transformed_df = pd.DataFrame({
        'transaction_id': range(1001, 1051),
        'transaction_date': pd.to_datetime(['2023-03-01'] * 50),
        'transaction_amount': [i * 1.1 for i in range(1, 51)],
        'currency_code': ['USD'] * 50,
        'merchant_name': [f'Processed Merchant {i%7}' for i in range(1,51)],
        'transaction_status': ['Processed'] * 50,
        'transactionyear': [2023] * 50,
        'transactionmonth': [3] * 50,
        'transactionday': [1] * 50,
        'transactiondayofweek': ['Wednesday'] * 50 # Adjust if date is different
    })
    
    # Ensure the target DB and table exist or can be created by to_sql.
    # For SQLite (example), it will create the DB file.
    # For PostgreSQL/MySQL, the database 'dw_financials' must exist, and user 'dw_user' needs permissions.
    # The table 'fact_transactions' will be created/replaced/appended to by to_sql.

    if config['target_database']['db_type'] == 'sqlite': # Easy for local testing
        db_path = os.path.join(project_r, 'data/output', config['target_database']['db_name'] + ".db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        config['target_database']['db_name'] = db_path # Override for sqlite path

    logger.info(f"Testing load with {config['target_database']['db_type']}")
    success = load_data_to_db(dummy_transformed_df, config)

    if success:
        logger.info("Load module test successful.")
        # You might want to add a check here to query the DB and verify data
    else:
        logger.error("Load module test failed.")
