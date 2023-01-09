import logging

from vxtools import summarize

# Set up logging at root level
from vxtools.logger import setup_logging
setup_logging()

log = logging.getLogger(__name__)