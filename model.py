import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# local_model_path = '/Users/keburius/Desktop/NLP/Models/ss-translation.pt'
# server_model_path = '/home/ubuntu/model/ss-translation.pt'


class TranslationModel:
    def __init__(self, model_repo='google/mt5-base', model_path='/home/ubuntu/model/ss-tr.pt'):
        self.LANG_TOKEN_MAPPING = {
            'DescriptionEn': '<en>',
            'DescriptionGe': '<ka>',
            'DescriptionRu': '<ru>',
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=False)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, config={'max_new_tokens': 128}).cuda()
        self.model.config.max_new_tokens = 128

        special_tokens_dict = {'additional_special_tokens': list(self.LANG_TOKEN_MAPPING.values())}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.load_state_dict(torch.load(model_path))

    def encode_input_str(self, text, target_lang, seq_len):
        target_lang_token = self.LANG_TOKEN_MAPPING[target_lang]

        input_ids = self.tokenizer.encode(
            text=target_lang_token + text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            use_fast=False
        )

        return input_ids[0]

    def translate_text(self, input_text, output_language):
        input_ids = self.encode_input_str(
            text=input_text,
            target_lang=output_language,
            seq_len=512,
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        output_tokens = self.model.generate(
            input_ids,
            num_beams=10,
            length_penalty=0.2
        ).cuda()
        predicted_translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return predicted_translation
