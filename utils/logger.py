#!/usr/bin/env python3
"""
Logging configuration for the Sign Language Translator application.
"""

import os
import logging
import logging.handlers
import datetime

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Setup application-wide logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # If no specific log file is provided, create one with timestamp
        if log_file == True:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/sign_translator_{timestamp}.log"
            os.makedirs("logs", exist_ok=True)
        
        # Configure rotating file handler (10 MB max size, keep 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {os.path.abspath(log_file)}")
    
    return logger

def set_log_level(level):
    """
    Change the logging level during runtime.
    
    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.info(f"Log level changed to {logging.getLevelName(level)}")