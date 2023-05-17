from flask import Flask, jsonify, request, render_template
from model import translate_text
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['text']
    output_language = request.form['target_lang']

    predicted_translation = translate_text(input_text, output_language)

    return jsonify({'translation': predicted_translation})


@app.route('/')
def index():
    return "Welcome to Ss-Translator"


if __name__ == '__main__':
    app.run()
