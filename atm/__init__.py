import os
import logging.config
from atm import _version

__version__ = _version.get_versions()["version"]

dirname = os.path.dirname(__file__)
try:
    logging.config.fileConfig(
        os.path.join(dirname, "logging.ini"), disable_existing_loggers=False
    )
except Exception:
    print("atm: Logging setup failed")
