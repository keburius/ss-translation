import torch
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer
import nltk

# model path for server ---> /home/ubuntu/model/ss-tr.pt
# model path for local  ---> /Users/keburius/Desktop/NLP/Models/ss-tr.pt


def split_text(text):
    point_added = False

    if not text.endswith('.'):
        text += '.'
        point_added = True

    sentences = nltk.sent_tokenize(text)
    parts = []
    current_part = ''

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if len(current_part.split()) + len(words) <= 50:
            current_part += ' ' + sentence
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = sentence

    if current_part:
        parts.append(current_part.strip())

    return parts, point_added


class TranslationModel:
    def __init__(self, model_repo='google/mt5-base', model_path='/Users/keburius/Desktop/NLP/Models/ss-tr.pt'):
        self.LANG_TOKEN_MAPPING = {
            'DescriptionEn': '<en>',
            'DescriptionGe': '<ka>',
            'DescriptionRu': '<ru>',
        }

        self.tokenizer = MT5Tokenizer.from_pretrained(model_repo)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, config={'max_new_tokens': 128}).to(self.device)
        self.model.config.max_new_tokens = 128

        special_tokens_dict = {'additional_special_tokens': list(self.LANG_TOKEN_MAPPING.values())}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def encode_input_str(self, text, target_lang, seq_len):
        target_lang_token = self.LANG_TOKEN_MAPPING[target_lang]

        input_ids = self.tokenizer.encode_plus(
            target_lang_token + text,
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            return_tensors='pt'
        )['input_ids']

        return input_ids[0]

    def translate_text(self, input_text, output_language):
        parts, point_added = split_text(input_text)
        translated_parts = []
        confidence_scores = []

        with torch.no_grad():
            for part in parts:
                input_ids = self.encode_input_str(
                    text=part,
                    target_lang=output_language,
                    seq_len=512,
                )
                input_ids = input_ids.unsqueeze(0).to(self.device)

                output_tokens = self.model.generate(
                    input_ids,
                    num_beams=10,
                    length_penalty=0.2
                ).to(self.device)

                predicted_translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                translated_parts.append(predicted_translation)

                tokenized_translation = self.tokenizer.encode(predicted_translation, add_special_tokens=False)
                translation_tensor = torch.tensor([tokenized_translation], dtype=torch.float32).to(self.device)
                confidence_scores.append(torch.softmax(translation_tensor, dim=1)[0].max().item())

        final_translation = ' '.join(translated_parts)
        overall_confidence = sum(confidence_scores) / len(confidence_scores)

        if point_added and final_translation.endswith('.'):
            final_translation = final_translation[:-1]

        return final_translation, overall_confidence
