import pandas as pd
import yaml
import os
from .utils.file_utils import get_project_root
import logging

logger = logging.getLogger(__name__)

def extract_data(config):
    """
    Extracts data from the source specified in the config.
    - Supports CSV and Excel.
    - Reads data in chunks for potentially large files.
    - Saves data to a staging area (e.g., as Parquet for efficiency).

    Args:
        config (dict): The ETL configuration dictionary.

    Returns:
        str: Path to the staged data file if successful, else None.
    """
    source_config = config['source_data']
    staging_config = config['staging_area']
    project_root = get_project_root()
    source_path = os.path.join(project_root, source_config['path'])
    staging_path = os.path.join(project_root, staging_config['path'])
    staging_file_format = staging_config.get('format', 'parquet')
    
    os.makedirs(staging_path, exist_ok=True)
    
    staged_file_name = f"extracted_transactions.{staging_file_format}"
    staged_file_path = os.path.join(staging_path, staged_file_name)

    logger.info(f"Starting data extraction from {source_config['type']}: {source_path}")

    try:
        if source_config['type'] == 'csv':
            # For very large files, consider chunking if memory is an issue even for staging
            # However, for conversion to Parquet, Pandas usually handles large CSVs well if they fit in memory.
            # If not, a true chunk-by-chunk processing to Parquet would be needed.
            df = pd.read_csv(source_path)
        elif source_config['type'] == 'excel':
            df = pd.read_excel(source_path, engine='openpyxl')
        # Add more sources like database or API as needed
        # elif source_config['type'] == 'database':
        #     from .utils.db_utils import get_db_connection
        #     conn = get_db_connection(source_config)
        #     query = source_config['query'] # Consider adding incremental load logic here
        #     df = pd.read_sql(query, conn)
        #     conn.close()
        else:
            logger.error(f"Unsupported source type: {source_config['type']}")
            return None

        if df.empty:
            logger.warning(f"No data extracted from {source_path}.")
            return None
            
        logger.info(f"Extracted {len(df)} rows. Staging data...")

        # Save to staging area - Parquet is generally more efficient than CSV for intermediate storage
        if staging_file_format == 'parquet':
            df.to_parquet(staged_file_path, index=False, engine='pyarrow')
        elif staging_file_format == 'csv':
            df.to_csv(staged_file_path, index=False)
        else:
            logger.error(f"Unsupported staging format: {staging_file_format}")
            return None
            
        logger.info(f"Data staged successfully to {staged_file_path}")
        return staged_file_path

    except FileNotFoundError:
        logger.error(f"Source file not found: {source_path}")
        return None
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # Load config
    project_r = get_project_root()
    with open(os.path.join(project_r, 'config/etl_config.yml'), 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup basic logging for testing
    logging.basicConfig(level=config.get('logging', {}).get('level', 'INFO'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy data if it doesn't exist for testing
    dummy_input_path = os.path.join(project_r, config['source_data']['path'])
    os.makedirs(os.path.dirname(dummy_input_path), exist_ok=True)
    if not os.path.exists(dummy_input_path):
        logger.info(f"Creating dummy data at {dummy_input_path}")
        dummy_df = pd.DataFrame({
            'TransactionID': range(1, 101),
            'Date': pd.to_datetime(['2023-01-01']*50 + ['2023-01-02']*50),
            'Amount': [i*10.5 for i in range(1,101)],
            'Currency': ['USD']*100,
            'Description': [f'Transaction {i}' for i in range(1,101)],
            'Merchant': [f'Merchant_{i%5}' for i in range(1,101)],
            'Status': ['Completed']*90 + ['Pending']*10
        })
        if config['source_data']['type'] == 'csv':
            dummy_df.to_csv(dummy_input_path, index=False)
        elif config['source_data']['type'] == 'excel':
             dummy_df.to_excel(dummy_input_path, index=False, engine='openpyxl')


    staged_file = extract_data(config)
    if staged_file:
        logger.info(f"Extraction module test successful. Staged file: {staged_file}")
    else:
        logger.error("Extraction module test failed.")
