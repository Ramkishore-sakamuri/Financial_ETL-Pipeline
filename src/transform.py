import pandas as pd
import logging
from .utils.file_utils import get_project_root
import os

logger = logging.getLogger(__name__)

def transform_data(staged_file_path, config):
    """
    Transforms the staged data.
    - Handles data cleaning, type conversion, new feature creation.
    - Uses Pandas for transformations.

    Args:
        staged_file_path (str): Path to the staged data file (e.g., Parquet from extract step).
        config (dict): The ETL configuration dictionary.

    Returns:
        pandas.DataFrame: The transformed DataFrame if successful, else None.
    """
    logger.info(f"Starting data transformation for {staged_file_path}")
    
    staging_format = config.get('staging_area', {}).get('format', 'parquet')
    transform_config = config.get('transformations', {})

    try:
        if staging_format == 'parquet':
            df = pd.read_parquet(staged_file_path)
        elif staging_format == 'csv':
            df = pd.read_csv(staged_file_path)
        else:
            logger.error(f"Unsupported staging format for reading: {staging_format}")
            return None

        logger.info(f"Read {len(df)} rows for transformation.")

        # --- Example Transformations for Financial Data ---
        # 1. Handle Missing Values
        # Example: Fill missing 'Merchant' with 'Unknown'
        if 'Merchant' in df.columns:
            df['Merchant'] = df['Merchant'].fillna('Unknown')
        # Example: For numerical columns, consider filling with 0 or mean/median, or dropping
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)

        # 2. Standardize Data Types
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # errors='coerce' will turn unparseable dates into NaT

        # 3. Data Cleaning
        # Example: Trim whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        # Example: Standardize 'Status' column
        if 'Status' in df.columns:
            df['Status'] = df['Status'].str.lower().replace({'completed': 'Processed', 'pending': 'AwaitingSettlement'})

        # 4. Create New Features (Domain-Specific)
        # Example: Extract Year, Month, Day from Transaction Date
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['TransactionYear'] = df['Date'].dt.year
            df['TransactionMonth'] = df['Date'].dt.month
            df['TransactionDay'] = df['Date'].dt.day
            df['TransactionDayOfWeek'] = df['Date'].dt.day_name()

        # Example: Categorize transactions based on amount
        # if 'Amount' in df.columns:
        #     bins = [0, 50, 200, 1000, float('inf')]
        #     labels = ['Small', 'Medium', 'Large', 'Very Large']
        #     df['AmountCategory'] = pd.cut(df['Amount'], bins=bins, labels=labels, right=False)

        # 5. Drop unnecessary columns (as per config)
        columns_to_drop = transform_config.get('drop_columns', [])
        if columns_to_drop:
            df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')
            logger.info(f"Dropped columns: {columns_to_drop}")

        # 6. Rename columns for consistency with target schema (if needed)
        # Example: rename_map = {"TransactionID": "transaction_id", "Amount": "transaction_amount"}
        # df.rename(columns=rename_map, inplace=True)

        logger.info(f"Transformation complete. DataFrame shape: {df.shape}")
        
        # --- Performance Note for Transformations ---
        # - Using vectorized operations in Pandas (as above) is crucial for performance.
        # - Avoid row-by-row iteration (df.iterrows(), df.apply(axis=1) with complex functions) where possible.
        # - For extremely large datasets not fitting in memory, consider tools like Dask or Spark.

        return df

    except Exception as e:
        logger.error(f"Error during transformation: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    import yaml
    # Example Usage (for testing this module directly)
    project_r = get_project_root()
    with open(os.path.join(project_r, 'config/etl_config.yml'), 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=config.get('logging', {}).get('level', 'INFO'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Assuming extract step has run and created a staging file
    staging_file_path = os.path.join(project_r, config['staging_area']['path'], f"extracted_transactions.{config['staging_area']['format']}")
    
    # Create dummy staging file if it doesn't exist
    if not os.path.exists(staging_file_path):
        logger.warning(f"Dummy staging file {staging_file_path} not found. Creating one for test.")
        os.makedirs(os.path.dirname(staging_file_path), exist_ok=True)
        dummy_staged_df = pd.DataFrame({
            'TransactionID': range(1, 51),
            'Date': pd.to_datetime(['2023-02-01']*25 + ['2023-02-02']*25),
            'Amount': [i*5.5 for i in range(1,51)],
            'Currency': ['USD']*50,
            'Description': [f' Raw Transaction {i} ' for i in range(1,51)],
            'Merchant': [f'Merchant_{i%3}' if i%3 != 0 else None for i in range(1,51)],
            'Status': ['Completed']*30 + ['Pending']*10 + ['completed']*10,
            'unused_column1': 'test'
        })
        if config['staging_area']['format'] == 'parquet':
            dummy_staged_df.to_parquet(staging_file_path, index=False)
        else:
            dummy_staged_df.to_csv(staging_file_path, index=False)


    transformed_df = transform_data(staging_file_path, config)

    if transformed_df is not None:
        logger.info("Transformation module test successful.")
        logger.info(f"Transformed data head:\n{transformed_df.head()}")
    else:
        logger.error("Transformation module test failed.")
