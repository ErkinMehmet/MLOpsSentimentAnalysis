
import logging
from pathlib import Path

# Crea la carpeta logs si no existe
Path("logs").mkdir(exist_ok=True)

def get_logger(name: str):
    logger = logging.getLogger(name)

    # Evitar agregar múltiples handlers si el logger ya está configurado
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # File handler → logs/<name>.log
    fh = logging.FileHandler(f"logs/{name}.log")
    fh.setLevel(logging.ERROR)

    # Formato
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger