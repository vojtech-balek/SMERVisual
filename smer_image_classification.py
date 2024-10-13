import base64

import pandas as pd
from openai import OpenAI, Embedding
import base64
from os import PathLike
from typing import Union
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class ImageClassifier:
    def __init__(self, data: Union[str, PathLike], use_openai=True, openai_model=None, openai_key=None,
                 local_model_path=None):
        """
        Initialize the ImageClassifier with OpenAI API or a local model.
        :param use_openai: Boolean flag to determine whether to use OpenAI API or a local model.
        :param openai_key: API key for OpenAI (required if use_openai is True).
        :param local_model_path: Custom local model object to be used (required if use_openai is False).
        """
        self.data_folder = data
        self.model_host = "openai" if use_openai else "local"
        if self.model_host == "openai":
            if not openai_key:
                raise ValueError("OpenAI key must be provided when using OpenAI API.")
            if not openai_model:
                raise ValueError("OpenAI model must be provided when using OpenAI API.")
            self.open_ai_key = openai_key
            self.open_ai_model = openai_model

        else:
            if not local_model_path:
                raise ValueError("Local model must be provided when not using OpenAI API.")

            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def __call__(self, file, prompt='Describe this image in 7 words.'):
        self.dataset = pd.DataFrame(getattr(self, f"_{self.model_host}_image_description")(file, prompt),
                                    columns=['Image, Description', 'Embedding'])
        self.dataset['Aggregated_embedding'] = self.dataset['Embedding'].apply(self._aggregate_embeddings)
        self.dataset.to_csv('embedded_dataset.csv')

    def _openai_image_description(self, prompt: str) -> Union[str, None]:
        """
        Generate image description with the use of OpenAI API.
        :param prompt: Text specification of the instruction for the LLM.
        :type prompt: str
        :return: Text description of the image if no exception occurs, None otherwise
        """
        self.client = OpenAI(api_key=self.open_ai_key)

        for file in self._get_image_files(self.data_folder):
            encoded_image = self.encode_image(file)
            try:
                response = self.client.chat.completions.create(
                    model=self.open_ai_model,
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
                yield file, response.choices[0].message.content, self._get_all_embeddings(response.choices[0].message.content)
            except Exception as e:
                print(f'Error: {e}')
                yield file, None, np.zeros(1536)

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

    def _get_all_embeddings(self, sentence: str) -> np.array:
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
                response = self.client.embeddings.create(
                    input=word,
                    model="text-embedding-ada-002"
                )
                embeddings.append(np.array(response))
            except Exception as e:
                print(f"Error generating embedding for '{word}': {e}")

        # If embeddings are successfully generated, return them; otherwise, return zeros
        if embeddings:
            return embeddings
        else:
            return np.zeros(1536)  # ADA embedding size is 1536

    def classify_with_logreg(self, df):

        # Extract features and labels
        X = np.stack(df['aggregated_embedding'].values)
        # y = df['Class'].apply(lambda x: 1 if x == 'vase' else 0)  # Binary encoding: vase=1, hotpot=0
        y = df['Class']

        train_size = 2000
        X_train = X[:train_size]
        X_test = X[train_size + 1:]
        X_test_embeddings = df['Embedding'][train_size + 1:].reset_index(drop=True)

        y_train = y[:train_size]
        y_test = y[train_size + 1:]
        # Split the data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print("Accuracy:", accuracy)
        df['Feature_Importance'] = None  # Initialize column
        df['Sorted_Words_by_Importance'] = None

        for idx in range(len(df)):
            # Get the original description
            description = df.Description[idx]

            # Get the original prediction probability
            original_texts = [description]
            original_embeddings = []
            words = description.split()
            word_indices = [description.split().index(word) for word in words if word in description.split()]
            print(word_indices)
            word_embeddings = [df['Embedding'].iloc[idx][i] for i in word_indices]

            if word_embeddings:
                # Aggregate the embeddings (e.g., by averaging)
                original_aggregated_embedding = np.mean(word_embeddings, axis=0)
            else:
                # Handle the case where no words are left (e.g., return a zero vector)
                original_aggregated_embedding = np.zeros_like(df['Embedding'].iloc[idx][0])

            original_embeddings.append(original_aggregated_embedding)
            original_embeddings = np.array(original_embeddings)

            # Ensure embeddings array is 2D (samples, features)
            if original_embeddings.ndim != 2 or original_embeddings.shape[1] != 1536:
                raise ValueError(f"Expected 2D array with 1536 features, got shape {original_embeddings.shape}")

            # Predict probabilities using the trained Logistic Regression model
            original_prob = model.predict_proba(original_embeddings)[0]

            # Prepare for storing feature importance scores
            importance_scores = {}

            # Remove each word one by one and calculate feature importance
            for word in words:
                print(words)
                # Remove the word from the description
                perturbed_text = ' '.join(w for w in words if w != word)
                print(perturbed_text)

                # Calculate the embeddings for the perturbed text
                perturbed_words = perturbed_text.split()
                print(perturbed_words)
                perturbed_word_indices = [description.split().index(w) for w in perturbed_words if
                                          w in description.split()]
                print(perturbed_word_indices)
                perturbed_word_embeddings = [df['Embedding'].iloc[idx][i] for i in perturbed_word_indices]
                print(len(perturbed_word_embeddings))

                if perturbed_word_embeddings:
                    # Aggregate the embeddings (e.g., by averaging)
                    perturbed_aggregated_embedding = np.mean(perturbed_word_embeddings, axis=0)
                else:
                    # Handle the case where no words are left (e.g., return a zero vector)
                    perturbed_aggregated_embedding = np.zeros_like(df['Embedding'].iloc[idx][0])

                perturbed_embeddings = np.array([perturbed_aggregated_embedding])

                # Ensure embeddings array is 2D (samples, features)
                if perturbed_embeddings.ndim != 2 or perturbed_embeddings.shape[1] != 1536:
                    raise ValueError(f"Expected 2D array with 1536 features, got shape {perturbed_embeddings.shape}")

                # Predict probabilities using the trained Logistic Regression model
                perturbed_prob = model.predict_proba(perturbed_embeddings)[0]

                # Calculate the drop in probability
                drop = original_prob - perturbed_prob

                # Adjust based on class
                target_class = df.Class[idx]
                if target_class == 'hotpot':
                    importance_scores[word] = drop[0]  # For hotpot, more negative means more important
                elif target_class == 'vase':
                    importance_scores[word] = -drop[0]  # For vase, more positive means more important

            # Sort words by importance score
            sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

            # Convert sorted importance_scores to a comma-separated string
            importance_str = ','.join(f"{word}:{score:.4f}" for word, score in sorted_importance)
            df.at[idx, 'Feature_Importance'] = importance_str

            # Create a list of words sorted by their importance
            sorted_words_list = [word for word, score in sorted_importance]
            sorted_words_str = ','.join(sorted_words_list)
            df.at[idx, 'Sorted_Words_by_Importance'] = sorted_words_str

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
    def _aggregate_embeddings(embedding_list):
        """Flatten the list of lists and take the mean along the axis"""
        return np.mean(embedding_list, axis=0)

    @staticmethod
    def _get_image_files(folder_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    yield os.path.join(root, file)


class ImageLabeler:
    """Label data using Large Language Model"""

    def __init__(self):
        pass
