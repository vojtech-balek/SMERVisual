"""smer_visual.py"""
from pathlib import Path
from tkinter import Image
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
from sklearn.model_selection import train_test_split
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from PIL import Image
import cv2
from typing import Union, Optional, List
from os import PathLike

from utils import _get_image_files_with_class, _encode_image


def image_description(
        model: Union[str, Path],
        data_folder: Union[str, Path],
        api_key: Optional[str] = None,
        user_prompt: str = "Describe this image in 7 words. Be concise, try to maximize the information about the objects in this image.",
) -> dict:
    """
    Generate image descriptions using either OpenAI or local models.

    Args:
        model: Model identifier (e.g., 'gpt-4o-mini', '4o', 'o3') or path to local model
        data_folder: Path to folder containing image files in class subfolders
        api_key: OpenAI API key (required for OpenAI models)
        user_prompt: Custom prompt for image description generation

    Returns:
        Dictionary mapping file paths to their descriptions, labels, and errors
    """
    OPENAI_MODELS = {'gpt-4o-mini', '4o', 'o3', 'o3-mini', 'o3-pro', 'o1-pro'}

    def process_with_openai(client: OpenAI) -> dict:
        results = {}
        for file_path, label in _get_image_files_with_class(data_folder):
            results[file_path] = {'label': label, 'description': None, 'error': None}
            try:
                encoded_image = _encode_image(file_path)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                        ]
                    }],
                    max_tokens=20
                )
                results[file_path]['description'] = response.choices[0].message.content
            except Exception as e:
                results[file_path]['error'] = str(e)
        return results

    def process_with_local_model() -> dict:
        results = {}
        try:
            processor = AutoProcessor.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained(model)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            local_model = MllamaForConditionalGeneration.from_pretrained(model).to(device)

            for file_path, label in _get_image_files_with_class(data_folder):
                results[file_path] = {'label': label, 'description': None, 'error': None}
                try:
                    image = Image.open(file_path)
                    message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": user_prompt}
                            ]
                        }
                    ]
                    input_text = processor.apply_chat_template(message, add_generation_prompt=True)
                    inputs = processor(
                        image,
                        input_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).to(device)

                    output = local_model.generate(**inputs, max_new_tokens=30)
                    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
                    results[file_path]['description'] = decoded_text.split('assistant\n\n')[1]

                    torch.cuda.empty_cache()
                    del output

                except Exception as e:
                    results[file_path]['error'] = str(e)

        except Exception as e:
            raise ValueError(f"Error initializing local model: {str(e)}")
        return results

    # Main logic
    if isinstance(model, str) and model in OPENAI_MODELS:
        if not api_key:
            raise ValueError("API key required for OpenAI models")
        return process_with_openai(OpenAI(api_key=api_key))
    else:
        return process_with_local_model()