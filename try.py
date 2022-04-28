import logging
logger = logging.getLogger(__name__.replace('_', ''))

from wind_scripts.utils import set_logger
# set_logger()
logger.info('info hello')
logger.warning('warn hello')