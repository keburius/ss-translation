from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import translate_text
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
app = Flask(__name__)
CORS(app)

@app.route('/translate')
def translate():
    input_text = request.args.get('text')
    output_language = request.args.get('target_lang')

    if input_text and output_language:
        predicted_translation = translate_text(input_text, output_language)
        return jsonify({'translation': predicted_translation})

    return jsonify({'error': 'Invalid request or missing data'})

@app.route('/')
def index():
    return "Welcome to Ss-Translator"

if __name__ == '__main__':
    app.run()
