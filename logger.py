import logging
import coloredlogs

logger = logging.getLogger('logger')
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt='%(asctime)s %(module)s.py:%(lineno)d %(levelname)s %(message)s')
