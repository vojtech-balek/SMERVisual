import os
import base64
import numpy as np
import spacy
import re


def _get_image_files_with_class(folder_path):
    """Load images with labels based on their parent directory name.
    :param folder_path: path to directory subfolders with images
    :type folder_path: str
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions) and '.ipynb_checkpoints' not in root:
                image_path = os.path.join(root, file)
                class_name = os.path.basename(root)

                yield image_path, class_name


def _encode_image(image_path: str):
    """
    Encode image to a base64 representation.
    :param image_path: Path to an image.
    :type image_path: str
    :return: base64 representation of the image
    """
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')


def _preprocess_text(text: str) -> str:
    """
    Preprocess the image descriptions.
    :param text: word description of the image
    :type text: string
    :return: lemmatized, normalized text
    """
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    words = text.split(',')
    lemmatized_words = []
    for word in words:
        word = word.strip()
        word = re.sub(r'[^\w\s]', '', word)
        if word:
            doc = nlp(word)
            lemma = ' '.join([token.lemma_ for token in doc])
            lemmatized_words.append(lemma)
    result = ','.join(lemmatized_words)
    return result
