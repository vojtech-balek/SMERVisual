import os
import base64
from openai import OpenAI, Embedding
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from os import PathLike


class ImageClassifier:
    def __init__(self, openai_key):
        self.open_ai_key = openai_key

    def image_description(self, file: Union[PathLike, str], prompt: str, model='gpt-4o-mini') -> Union[str, None]:
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
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
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
