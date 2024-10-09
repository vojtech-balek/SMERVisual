import os
import base64
from openai import OpenAI, Embedding
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from os import PathLike
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ast
import spacy
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot  as plt
import seaborn as sns
import re
from tqdm import tqdm

class ImageClassifier:
    def __init__(self, use_openai=True, openai_key=None, local_model_path=None):
        """
        Initialize the ImageClassifier with OpenAI API or a local model.
        :param use_openai: Boolean flag to determine whether to use OpenAI API or a local model.
        :param openai_key: API key for OpenAI (required if use_openai is True).
        :param local_model_path: Custom local model object to be used (required if use_openai is False).
        """
        self.use_openai = use_openai
        if self.use_openai:
            if not openai_key:
                raise ValueError("OpenAI key must be provided when using OpenAI API.")
            self.open_ai_key = openai_key

        else:
            if not local_model_path:
                raise ValueError("Local model must be provided when not using OpenAI API.")

            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def _openai_image_description(self, file: Union[PathLike, str], prompt: str, model='gpt-4o-mini') -> Union[str, None]:
        """
        Generate image description with the use of OpenAI API.
        :param file: path to a jpeg file
        :type file: PathLike or str
        :param prompt: Text specification of the instruction for the LLM.
        :type prompt: str
        :param model: OpenAI model to use
        :type model: str
        :return: Text description of the image if no exception occurs, None otherwise
        """
        client = OpenAI(api_key=self.open_ai_key)

        encoded_image = self.encode_image(file)
        try:
            # Create the GPT-4o API request
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url":
                                    {
                                        "url": f"data:image/png;base64,{encoded_image}"
                                    }
                            },
                        ],
                    }
                ],
                max_tokens=20,
            )

            # Extract and return the description
            return response.choices[0].message.content
        except Exception as e:
            print(f'Error: {e}')
            return None

    def _local_image_description(self, file: Union[PathLike, str], prompt: str) -> Union[str, None]:
        """
        Generate image description using the local Hugging Face transformer model.
        :param file: Path to the image file.
        :param prompt: Text prompt for the model.
        :return: Generated description.
        """
        try:
            encoded_image = self.encode_image(file)

            input_text = f"{prompt} Image: {encoded_image}"

            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            outputs = self.model.generate(inputs.input_ids, max_length=50)

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f'Error: {e}')
            return None

    # Function to transform sentence to embeddings and compute average embeddings
    @staticmethod
    def get_all_embeddings(sentence: str) -> np.array:
        """
        Generate embeddings for the image's word description.
        :param sentence: Word description of an image.
        :type sentence: str
        :return: Embedding of length 1536
        """
        words = sentence.split()
        embeddings = []

        for word in words:
            try:
                response = Embedding.create(
                    input=word,
                    model="text-embedding-ada-002"
                )
                word_embedding = response['data'][0]['embedding']
                embeddings.append(np.array(word_embedding))
            except Exception as e:
                print(f"Error generating embedding for '{word}': {e}")

        # If embeddings are successfully generated, return them; otherwise, return zeros
        if embeddings:
            return embeddings
        else:
            return np.zeros(1536)  # ADA embedding size is 1536

    def classify_with_logreg(self):
        pass
    @staticmethod
    def encode_image(image_path: str):
        """
        Encode image to a base64 representation.
        :param image_path: Path to an aimage.
        :type image_path: str
        :return: base64 representation of the image
        """
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')

    @staticmethod
    def aggregate_embeddings(embedding_list):
        """Flatten the list of lists and take the mean along the axis"""
        return np.mean(embedding_list, axis=0)
