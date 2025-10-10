import logging
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for WhisperX.

    Args:
        level: Logging level (debug, info, warning, error, critical). Default: info
        log_file: Optional path to log file. If None, logs only to console.
    """
    logger = logging.getLogger("whisperx")

    logger.handlers.clear()

    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.WARNING
    logger.setLevel(log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError) as e:
            logger.warning(f"Failed to create log file '{log_file}': {e}")
            logger.warning("Continuing with console logging only")

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with WhisperX settings
    """
    whisperx_logger = logging.getLogger("whisperx")
    if not whisperx_logger.handlers:
        setup_logging()

    logger_name = "whisperx" if name == "__main__" else name
    return logging.getLogger(logger_name)
