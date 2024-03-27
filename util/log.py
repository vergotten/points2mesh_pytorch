import logging
import os
import sys
import time


# Append the parent directory to the system path
sys.path.append('../')

# Import the configuration object from the util module
from util.config import cfg

def create_logger(log_file):
    """
    Creates a logger for logging debug information.

    Parameters:
    log_file (str): The file to which the log should be written.

    Returns:
    logger: A logging.Logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)    # filename: build a FileHandler
    return logger

# Determine the log file name based on the task type
if cfg.task == 'train':
    log_file = os.path.join(
        cfg.exp_path,
        'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
elif cfg.task == 'test':
    log_file = os.path.join(
        cfg.exp_path, cfg.result_path, 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
        cfg.split, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
elif cfg.task == 'inference':
    log_file = os.path.join(
        cfg.exp_path, cfg.result_path, 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
        cfg.split, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
# Create the directory for the log file if it doesn't exist
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Create the logger
logger = create_logger(log_file)

# Log the start of the logging
logger.info('************************ Start Logging ************************')