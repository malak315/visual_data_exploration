"""
Logging utilities for structured logging to files and console.

This module provides a centralized logging system that can log to both files
and console, with structured output that can be easily analyzed later.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


from lib.config import LOGS_DIR, LOG_LEVEL, LOG_TO_FILE, LOG_TO_CONSOLE


class StructuredLogger:
    """
    A structured logger that writes to both files and console.
    
    Logs are written in a structured format that can be easily parsed
    and analyzed, making it suitable for feeding into AI analysis tools.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        level: str = "INFO"
    ):
        """
        Initialize the structured logger.
        
        Args:
            name: Name of the logger (typically module name)
            log_dir: Directory to save log files (defaults to config.LOGS_DIR)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.log_dir = log_dir or LOGS_DIR
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        
        # File handler
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def section(self, title: str):
        """Log a section header for better organization."""
        separator = "=" * 80
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}")
    
    def subsection(self, title: str):
        """Log a subsection header."""
        separator = "-" * 80
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}")
    
    def metric(self, name: str, value, unit: str = ""):
        """Log a metric in a structured format."""
        if unit:
            self.info(f"METRIC: {name} = {value} {unit}")
        else:
            self.info(f"METRIC: {name} = {value}")
    
    def table(self, title: str, data: dict):
        """Log a table of key-value pairs."""
        self.info(f"\n{title}:")
        for key, value in data.items():
            self.info(f"  {key}: {value}")
    
    def list(self, title: str, items: list):
        """Log a list of items."""
        self.info(f"\n{title}:")
        for i, item in enumerate(items, 1):
            self.info(f"  {i}. {item}")


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "xai_analysis") -> StructuredLogger:
    """
    Get or create the global logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        StructuredLogger instance
    """
    global _logger
    if _logger is None:
        _logger = StructuredLogger(
            name=name,
            log_to_file=LOG_TO_FILE,
            log_to_console=LOG_TO_CONSOLE,
            level=LOG_LEVEL
        )
    return _logger


def reset_logger():
    """Reset the global logger (useful for testing)."""
    global _logger
    _logger = None

