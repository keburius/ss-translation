# api_key_generator.py

import secrets


def generate_api_key():
    return secrets.token_hex(16)
