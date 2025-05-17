# File: src/__init__.py
# This file makes the 'src' directory a Python package.
# You can add package-level initializations here if needed.

# For example, you might want to make certain utility functions or constants
# available directly when 'src' is imported, though it's often kept empty
# or used for more specific package setup.

# print("src package initialized") # Example initialization
```python
# File: src/extract/__init__.py
# This file makes the 'extract' directory a Python subpackage of 'src'.
# It allows you to import modules like: from src.extract import file_extractor

# You can define what is imported when 'from src.extract import *' is used:
# __all__ = ['file_extractor', 'api_extractor', 'db_extractor']

# Or, you can import specific modules to make them available at the package level:
# from .file_extractor import FileExtractor
# from .api_extractor import ApiExtractor

# For now, it can be kept empty.
```python
# File: src/transform/__init__.py
# This file makes the 'transform' directory a Python subpackage of 'src'.
# It allows you to import modules like: from src.transform import data_cleaner

# Similar to other __init__.py files, you can control imports or add initialization code.
# __all__ = ['data_cleaner', 'transaction_transformer', 'schema_mapper']
```python
# File: src/load/__init__.py
# This file makes the 'load' directory a Python subpackage of 'src'.
# It allows you to import modules like: from src.load import db_loader

# __all__ = ['db_loader', 'file_loader']
```python
# File: src/utils/__init__.py
# This file makes the 'utils' directory a Python subpackage of 'src'.
# It allows you to import modules like: from src.utils import helpers

# __all__ = ['helpers', 'performance_logger']
