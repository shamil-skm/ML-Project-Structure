import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
try:
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
except Exception as e:
    logging.error(f"Error creating logs directory: {e}")
logs_path = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=logs_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)


if __name__ == "__main__":
    logging.info("Logging has started")