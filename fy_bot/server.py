"""
Flask API Entrypoint
"""

from flask import Flask

from fy_bot.controller import message_blueprint

app = Flask(__name__)

app.register_blueprint(message_blueprint, url_prefix="/message")

if __name__ == "__main__":
    app.run(debug=True)
