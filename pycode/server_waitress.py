
from server_flask_app import app as flask_app
from waitress import serve

if __name__ == '__main__':
    serve(flask_app, host='0.0.0.0', port=9696)
