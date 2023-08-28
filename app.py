import os, sys
from config.setting import SERVER_PORT
from api.topic import app

# root
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)  # root path save to temp env

if __name__ == '__main__':
    # debug=true dev mode
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True)