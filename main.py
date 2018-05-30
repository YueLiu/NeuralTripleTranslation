from flask import Flask, request
from flask import render_template
from flask_cors import CORS
import get_model_api
from ui import app

if __name__ == '__main__':
    app.app.run(host='0.0.0.0', debug=True)
