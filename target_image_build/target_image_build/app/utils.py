import logging
import logging.handlers
import re

import coloredlogs


def create_logger(logger_name):
    # Create Logger
    logger = logging.getLogger(logger_name)

    # Check handler exists
    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = coloredlogs.ColoredFormatter(
        "%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%f",
        field_styles={
            "levelname": {"color": 120, "bold": True},
            "name": {"color": 200, "bold": False},
            "process": {"color": "cyan"},
            "asctime": {"color": 240},
            "message": {"color": 231},
        },
        level_styles={
            "debug": {"color": "green"},
            "verbose": {"color": "green", "bright": True},
            "info": {"color": "cyan", "bright": True},
            "notice": {"color": "cyan", "bold": True},
            "warning": {"color": "yellow"},
            "error": {"color": "red", "bright": True},
            "success": {"color": 77},
            "critical": {"background": "red", "color": 255, "bold": True},
        },
    )
    # Create Handlers
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    return logger


def clear_color_char(text):
    ansi_escape = re.compile(r"\'|(\x1b\[[\d;]+[m|E|\:|\s]*)|\'|\"")
    result = ansi_escape.sub("", text)
    return result
