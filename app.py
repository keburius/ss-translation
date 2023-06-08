from flask import Flask, jsonify, request
from flask_cors import CORS
from model import TranslationModel

app = Flask(__name__)
CORS(app)

# Initialize the TranslationModel class and keep it in memory.
translation_model = TranslationModel()


@app.route('/translate', methods=['POST'])
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
    app.run()
