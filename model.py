import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
model_repo = 'google/mt5-base'
# model_path = '/Users/keburius/Desktop/NLP/Models/ss-translation.pt'
model_path = 'ss-translation.pt'
max_seq_len = 128

LANG_TOKEN_MAPPING = {
    'DescriptionEn': '<en>',
    'DescriptionGe': '<ka>',
    'DescriptionRu': '<ru>',
}

# Implement tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, config={'max_new_tokens': 128}).to(device)
model.config.max_new_tokens = 128

special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# model.load_state_dict(torch.load(model_path, map_location=device))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# Encoder
def encode_input_str(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
    target_lang_token = lang_token_map[target_lang]

    # Tokenize and add special tokens
    input_ids = tokenizer.encode(
        text=target_lang_token + text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len,
        use_fast=False  # Disable fast tokenization
    )

    return input_ids[0]


def translate_text(input_text, output_language):
    input_ids = encode_input_str(
        text=input_text,
        target_lang=output_language,
        tokenizer=tokenizer,
        seq_len=512,
        lang_token_map=LANG_TOKEN_MAPPING)
    input_ids = input_ids.unsqueeze(0).to(device)

    output_tokens = model.generate(
        input_ids,
        num_beams=20,  # Configure the number of beams
        length_penalty=0.2  # Configure the length penalty
    )
    predicted_translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return predicted_translation
