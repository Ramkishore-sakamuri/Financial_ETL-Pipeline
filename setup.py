# setup.py

from setuptools import setup, find_packages
import os

# Function to read the README file for long description
def read_readme():
    """Reads the README file for the long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "An ETL pipeline for processing financial transaction data, designed for improved data load times."

# Define core dependencies
# Specific database drivers (psycopg2-binary, pyodbc) can be made optional
# or installed separately by the user depending on their database.
# For a library, it's often better to keep these out of core install_requires
# or list them as extras.
# For an application, you might include the specific ones you use.
core_dependencies = [
    'pandas>=1.3.0,<3.0.0',    # For data manipulation
    'PyYAML>=5.4.0,<7.0.0',     # For loading YAML configurations
    'requests>=2.25.0,<3.0.0',  # For API extraction
    'SQLAlchemy>=1.4.0,<3.0.0', # For database interaction (especially with DbLoader)
    # Add other core libraries your project directly depends on to run
]

# Optional dependencies for specific databases or features
# Users can install these like: pip install financial-etl-pipeline[postgresql]
extras_require = {
    'postgresql': ['psycopg2-binary>=2.9.0,<3.0.0'],
    'sqlserver': ['pyodbc>=4.0.0,<5.0.0'],
    'parquet': ['pyarrow>=7.0.0,<16.0.0'], # For FileLoader/FileExtractor Parquet support
    'dev': [ # Development and testing dependencies
        'pytest>=6.2.0,<9.0.0',
        'requests-mock>=1.9.0,<2.0.0',
        'flake8>=3.9.0,<8.0.0',
        'mypy>=0.900,<2.0.0', # Optional: for static type checking
        # Add other dev tools like black, isort, coverage etc.
    ]
}
extras_require['all'] = sum(extras_require.values(), []) # To install all optional dependencies

setup(
    name='financial_etl_pipeline',
    version='0.1.0',  # Initial version
    author='Your Name / Your Organization', # Replace with your name/org
    author_email='your.email@example.com', # Replace with your email
    description='ETL pipeline for financial transaction data with focus on performance.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/financial-etl-pipeline',  # Replace with your project's GitHub URL
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/financial-etl-pipeline/issues', # Replace
    },
    license='MIT',  # Or choose another license like Apache 2.0, GPLv3, etc.
    
    # find_packages will discover all packages in the 'src' directory
    # It looks for directories with an __init__.py file.
    packages=find_packages(where='src'),
    
    # package_dir tells setuptools that packages are under 'src'
    package_dir={'': 'src'},
    
    # List of dependencies
    install_requires=core_dependencies,
    
    # Optional dependencies
    extras_require=extras_require,
    
    # Minimum Python version required
    python_requires='>=3.8', # Specify your target Python version
    
    # Classifiers help users find your project by browsing PyPI
    # Full list: https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',  # Or 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Database :: Front-Ends',
        'Topic :: Office/Business :: Financial',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License', # Change if you use a different license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Natural Language :: English',
    ],
    
    keywords='etl, financial data, data pipeline, performance, transactions', # Keywords for PyPI search

    # If your project has command-line scripts, define them here
    # For example, if you have a main.py that can be run from the command line:
    # entry_points={
    #     'console_scripts': [
    #         'run-financial-etl=financial_etl_pipeline.main:main_cli_function', # Assuming main.py has main_cli_function
    #     ],
    # },

    # If you have data files that need to be included with your package (e.g., default configs)
    # package_data={
    #     'your_package_name': ['config/*.yaml'], # Example: include all .yaml files in a config subfolder
    # },
    # include_package_data=True, # Usually used with MANIFEST.in for more complex data file needs
)
