import json
from datetime import datetime

def log(data):
    log_entry = {
        "timestamp": str(datetime.now()),
        "data": data
    }

    with open("logs.txt", "a") as f:
        f.write(json.dumps(log_entry) + "\n")