from flask import Flask
from api import api
from flask_cors import CORS
import threading
from drehgeberController import run_drehgeber

app = Flask(__name__)
CORS(app)
app.register_blueprint(api)

if __name__ == '__main__':
    
    # Drehgeber in separatem Thread starten ✅
    drehgeber_thread = threading.Thread(target=run_drehgeber, daemon=True)
    drehgeber_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
