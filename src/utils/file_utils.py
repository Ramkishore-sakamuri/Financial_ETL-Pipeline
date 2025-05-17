import os

def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    # Assuming this file is in src/utils/
    # Adjust the number of os.path.dirname calls if the file structure changes
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add other file-related utilities here if needed, e.g.,
# - finding specific files
# - cleaning up directories
