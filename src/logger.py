import logging
import os
from datetime import datetime

def setup_logger(name='video_chatbot'):
    """
    Setup comprehensive logging configuration
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log filename
    log_filename = os.path.join(
        log_dir, 
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Global logger
logger = setup_logger()