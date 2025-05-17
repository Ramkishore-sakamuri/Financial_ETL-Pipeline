from sqlalchemy import create_engine, exc
import logging

logger = logging.getLogger(__name__)

def get_db_engine(db_config):
    """
    Creates and returns a SQLAlchemy engine based on the database configuration.

    Args:
        db_config (dict): Database configuration dictionary containing keys like
                          'db_type', 'db_user', 'db_password', 'db_host',
                          'db_port', 'db_name'.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine object, or None if connection fails.
    """
    try:
        db_type = db_config['db_type'].lower()
        user = db_config['db_user']
        password = db_config['db_password']
        host = db_config['db_host']
        port = db_config['db_port']
        dbname = db_config['db_name']

        if db_type == 'postgresql':
            conn_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == 'mysql':
            conn_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == 'mssql':
            # Requires pyodbc and an ODBC driver like "ODBC Driver 17 for SQL Server"
            # conn_string = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{dbname}?driver=ODBC+Driver+17+for+SQL+Server"
             conn_string = f"mssql+pyodbc://{user}:{password}@{host}/{dbname}?driver=SQL+Server" # Simpler for local/default instances
        elif db_type == 'sqlite':
            # For SQLite, dbname is the path to the .db file. Host, port, user, pass are ignored.
            conn_string = f"sqlite:///{dbname}"
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return None

        engine = create_engine(conn_string, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as connection:
            logger.info(f"Successfully connected to {db_type} database: {dbname} on {host}")
        return engine

    except KeyError as e:
        logger.error(f"Database configuration missing key: {e}")
        return None
    except exc.SQLAlchemyError as e: # Catch SQLAlchemy specific errors for better diagnosis
        logger.error(f"Error creating database engine or connecting to {db_config.get('db_type', 'DB')}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating DB engine: {e}", exc_info=True)
        return None

# You could add more utility functions here, e.g., for executing custom SQL,
# bulk loading with database-specific tools if Pandas to_sql isn't optimal for some edge case.
