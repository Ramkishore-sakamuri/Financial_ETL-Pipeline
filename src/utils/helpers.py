# src/utils/helpers.py

import yaml
import logging
import datetime
import uuid
from typing import Dict, Any, Optional

# Get a logger for this module
# This will inherit the configuration from the root logger or a specific app logger
# if logging is configured at the application entry point.
logger = logging.getLogger(__name__)

def load_yaml_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely loads a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the configuration,
                                  or None if an error occurs.
    """
    logger.debug(f"Attempting to load YAML configuration from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML configuration from: {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"YAML configuration file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading YAML {file_path}: {e}", exc_info=True)
        return None

def get_formatted_timestamp(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Returns the current timestamp formatted as a string.

    Args:
        fmt (str): The strftime format string.
                   Defaults to "%Y-%m-%d_%H-%M-%S".

    Returns:
        str: The formatted timestamp string.
    """
    now = datetime.datetime.now()
    formatted_ts = now.strftime(fmt)
    logger.debug(f"Generated formatted timestamp: {formatted_ts} with format '{fmt}'")
    return formatted_ts

def generate_run_id(prefix: str = "run") -> str:
    """
    Generates a unique run ID, typically combining a prefix and a UUID.

    Args:
        prefix (str): A prefix for the run ID.

    Returns:
        str: A unique run ID string.
    """
    unique_part = str(uuid.uuid4().hex)[:8] # Use a portion of a UUID for brevity
    run_id = f"{prefix}_{get_formatted_timestamp('%Y%m%d%H%M%S')}_{unique_part}"
    logger.info(f"Generated Run ID: {run_id}")
    return run_id

def clean_column_names(df, case='snake'):
    """
    Cleans DataFrame column names.
    Converts to snake_case by default or other cases.

    Args:
        df (pd.DataFrame): Input DataFrame.
        case (str): 'snake' for snake_case, 'lower' for lowercase, 'upper' for uppercase.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    new_columns = []
    for col in df.columns:
        col = str(col).strip().replace(' ', '_').replace('-', '_').replace('.', '_')
        # Remove special characters except underscore
        col = ''.join(e for e in col if e.isalnum() or e == '_')
        # Handle multiple underscores
        col = '_'.join(filter(None, col.split('_')))

        if case == 'snake':
            # Basic snake_case conversion (can be improved for complex camelCase)
            col = col.lower()
        elif case == 'lower':
            col = col.lower()
        elif case == 'upper':
            col = col.upper()
        new_columns.append(col)
    df.columns = new_columns
    logger.info(f"Cleaned column names to {case} case. New columns: {df.columns.tolist()}")
    return df


if __name__ == '__main__':
    # Configure basic logging for testing this module directly
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing helper functions ---")

    # Test YAML loading (requires a dummy YAML file)
    dummy_yaml_path = "dummy_config_helpers.yaml"
    dummy_config_content = {
        'setting1': 'value1',
        'database': {
            'host': 'localhost',
            'port': 5432
        },
        'feature_flags': [True, False, True]
    }
    try:
        with open(dummy_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dummy_config_content, f)
        logger.info(f"Created dummy YAML file: {dummy_yaml_path}")

        loaded_config = load_yaml_config(dummy_yaml_path)
        if loaded_config:
            logger.info(f"Successfully loaded dummy config: {loaded_config}")
            assert loaded_config['database']['host'] == 'localhost'
        else:
            logger.error("Failed to load dummy config.")

        # Test loading a non-existent file
        logger.info("\nTesting loading non-existent YAML...")
        non_existent_config = load_yaml_config("non_existent.yaml")
        assert non_existent_config is None

    except Exception as e:
        logger.error(f"Error during YAML test setup or execution: {e}")
    finally:
        import os
        if os.path.exists(dummy_yaml_path):
            os.remove(dummy_yaml_path)
            logger.info(f"Removed dummy YAML file: {dummy_yaml_path}")

    # Test timestamp generation
    logger.info("\n--- Testing timestamp generation ---")
    ts1 = get_formatted_timestamp()
    logger.info(f"Default timestamp: {ts1}")
    ts2 = get_formatted_timestamp(fmt="%A, %B %d, %Y - %I:%M:%S %p")
    logger.info(f"Custom format timestamp: {ts2}")
    assert ts1 is not None and ts2 is not None

    # Test run ID generation
    logger.info("\n--- Testing run ID generation ---")
    run_id1 = generate_run_id()
    logger.info(f"Generated run_id (default prefix): {run_id1}")
    run_id2 = generate_run_id(prefix="etl_job")
    logger.info(f"Generated run_id (custom prefix): {run_id2}")
    assert run_id1.startswith("run_")
    assert run_id2.startswith("etl_job_")
    assert run_id1 != run_id2

    # Test column name cleaning
    logger.info("\n--- Testing column name cleaning ---")
    try:
        import pandas as pd # Import pandas only if testing this part
        data = {'First Name': ['Alice', 'Bob'], 'Last-Name': ['Smith', 'Doe'], 'Age.In.Years': [20,30]}
        df_test_cols = pd.DataFrame(data)
        logger.info(f"Original columns: {df_test_cols.columns.tolist()}")
        df_cleaned_snake = clean_column_names(df_test_cols.copy(), case='snake')
        logger.info(f"Snake case columns: {df_cleaned_snake.columns.tolist()}")
        assert df_cleaned_snake.columns.tolist() == ['first_name', 'last_name', 'age_in_years']

        df_cleaned_upper = clean_column_names(df_test_cols.copy(), case='upper')
        logger.info(f"Upper case columns: {df_cleaned_upper.columns.tolist()}")
        assert df_cleaned_upper.columns.tolist() == ['FIRST_NAME', 'LAST_NAME', 'AGE_IN_YEARS']

    except ImportError:
        logger.warning("Pandas not installed. Skipping column name cleaning test.")
    except Exception as e:
        logger.error(f"Error during column cleaning test: {e}")


    logger.info("\nHelper function tests finished.")
