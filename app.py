import json
from flask import Flask, jsonify, request
from flask_httpauth import HTTPTokenAuth
from api_key_generator import generate_api_key
from model import TranslationModel
from flask_cors import CORS


# Load existing keys from .env file
with open('.env', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        name, val = line.split('=', 1)
        if name == 'KEYS':
            keys = json.loads(val)


app = Flask(__name__)
CORS(app)

# Initialize the TranslationModel class and keep it in memory.
translation_model = TranslationModel()

# Initialize the HTTPTokenAuth object.
auth = HTTPTokenAuth(scheme='Bearer')


@auth.verify_token
def verify_token(token):
    if token in keys.values():
        return token  # Return the token itself as the authentication token
    return None


@app.route('/generate-api-key', methods=['POST'])
def new_api_key():
    username = request.json.get('user')
    if username is None:
        return jsonify({'error': 'Missing user identifier'}), 400

    # Check if this username already has an API key
    if username in keys:
        return jsonify({'error': 'Username already exists'}), 400

    new_key = generate_api_key()

    # Update the keys and rewrite the .env file
    keys[username] = new_key
    with open('.env', 'w') as f:
        f.write(f'KEYS={json.dumps(keys)}')

    return jsonify({'api_key': new_key}), 201


@app.route('/translate', methods=['POST'])
@auth.login_required
def translate():
    input_text = request.json.get('text')
    output_language = request.json.get('target_lang')

    if input_text and output_language:
        predicted_translation, confidence = translation_model.translate_text(input_text, output_language)
        return jsonify({'translation': predicted_translation, 'confidence': confidence})

    return jsonify({'error': 'Invalid request or missing data'})


@app.route('/')
def index():
    return "Welcome to Ss-Translator"


if __name__ == '__main__':
    # app.run()
    app.run(debug=True)
