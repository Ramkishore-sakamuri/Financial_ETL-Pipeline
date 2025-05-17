import yaml
import logging
import os
import argparse
from datetime import datetime

from .extract import extract_data
from .transform import transform_data
from .load import load_data_to_db
from .utils.file_utils import get_project_root

# --- Logging Setup ---
# Basic configuration. For production, consider a more robust setup (e.g., rotating file handlers).
def setup_logging(config):
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file')
    project_root = get_project_root()

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()] # Log to console by default

    if log_file:
        full_log_path = os.path.join(project_root, log_file)
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)
        handlers.append(logging.FileHandler(full_log_path))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logger = logging.getLogger(__name__) # Get logger for this module
    logger.info("Logging configured.")
    return logger


def run_pipeline(config_path):
    """
    Orchestrates the ETL pipeline: Extract, Transform, Load.
    """
    project_root = get_project_root()
    full_config_path = os.path.join(project_root, config_path)

    if not os.path.exists(full_config_path):
        print(f"ERROR: Configuration file not found at {full_config_path}")
        return

    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config) # Logger is now configured and available

    logger.info("=== Starting ETL Pipeline Run ===")
    start_time = datetime.now()

    # 1. Extract
    logger.info("--- Stage 1: Extract ---")
    staged_file_path = extract_data(config)
    if not staged_file_path:
        logger.error("Extraction failed. Aborting pipeline.")
        return
    logger.info(f"Extraction successful. Staged data at: {staged_file_path}")

    # 2. Transform
    logger.info("--- Stage 2: Transform ---")
    transformed_df = transform_data(staged_file_path, config)
    if transformed_df is None:
        logger.error("Transformation failed. Aborting pipeline.")
        return
    logger.info(f"Transformation successful. {len(transformed_df)} rows transformed.")

    # 3. Load
    logger.info("--- Stage 3: Load ---")
    load_success = load_data_to_db(transformed_df, config)
    if not load_success:
        logger.error("Load failed. Pipeline did not complete successfully.")
        return
    logger.info("Load successful.")

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"=== ETL Pipeline Run Finished Successfully in {duration} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Financial ETL Pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/etl_config.yml",
        help="Path to the ETL configuration YAML file (relative to project root)."
    )
    args = parser.parse_args()
    
    run_pipeline(args.config)
