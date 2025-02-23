
### 3. `utils/helpers.py`

# ```python
import logging
import json

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    handler = logging.FileHandler(log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def parse_config(config_file):
    """Function to parse configuration file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config
