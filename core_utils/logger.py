import logging


def setup_logger(level=logging.INFO):
    """Configure and return a project logger."""
    logger = logging.getLogger('isl_sign_recognition')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()
