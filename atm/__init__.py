import os
import logging.config
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("func2argparse")
except PackageNotFoundError:
    pass


dirname = os.path.dirname(__file__)
try:
    logging.config.fileConfig(
        os.path.join(dirname, "logging.ini"), disable_existing_loggers=False
    )
except Exception:
    print("atm: Logging setup failed")
