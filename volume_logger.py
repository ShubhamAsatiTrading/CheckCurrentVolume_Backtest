# volume_logger.py - Dedicated logging for volume operations
import os
import logging
from datetime import datetime

class VolumeLogger:
    _logger = None
    _log_file = None
    
    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._setup_logger()
        return cls._logger
    
    @classmethod
    def _setup_logger(cls):
        cls._log_file = f"volume_operations_{datetime.now().strftime('%Y%m%d')}.log"
        cls._logger = logging.getLogger('VolumeOperations')
        cls._logger.setLevel(logging.INFO)
        
        if not cls._logger.handlers:
            handler = logging.FileHandler(cls._log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)
    
    @classmethod
    def log(cls, message, level="INFO"):
        logger = cls.get_logger()
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    @classmethod
    def get_log_file_path(cls):
        if cls._log_file is None:
            cls._setup_logger()
        return cls._log_file