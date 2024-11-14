# read version from installed package
import logging
from importlib.metadata import version

__version__ = "1.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
