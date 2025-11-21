import logging
import os

logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, "running_log.log")

if os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as f:
        f.truncate(0)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",  
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)