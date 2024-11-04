import base64
from tkinter import Image

import pandas as pd
from openai import OpenAI, Embedding
import base64
from os import PathLike
from typing import Union
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import MllamaForConditionalGeneration, AutoProcessor

import os
from sklearn.model_selection import train_test_split
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image


class ImageClassifier:
    def __init__(self, openai_model=None, openai_embedding=None, openai_key=None,
                 local_model_path=None, local_embedding_path=None):
        """
        Initialize the ImageClassifier with OpenAI API or a local model.
        :param use_openai: Boolean flag to determine whether to use OpenAI API or a local model.
        :param openai_key: API key for OpenAI (required if use_openai is True).
        :param local_model_path: Custom local model object to be used (required if use_openai is False).
        """
        self.df_AOPC = None
        self.logreg_model = None
        self.model_host = "openai" if openai_model else "local"
        if self.model_host == "openai":
            if not openai_key:
                raise ValueError("OpenAI key must be provided when using OpenAI API.")
            if not openai_model:
                raise ValueError("OpenAI model must be provided when using OpenAI API.")
            self.open_ai_key = openai_key
            self.open_ai_model = openai_model
            self.openai_embedding = openai_embedding

        else:
            if not local_model_path:
                raise ValueError("Local model must be provided when not using OpenAI API.")

            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.local_embedding = local_embedding_path
            self.model = MllamaForConditionalGeneration.from_pretrained(
                local_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(local_model_path)

    def __call__(self, data: Union[str, PathLike],
                 prompt='You are a helpful assistant to help users describe images.', ):
        self.data_folder = data
        self.dataset = pd.DataFrame(getattr(self, f"_{self.model_host}_image_description")(prompt),
                                    columns=['Image', 'Description', 'Embedding', 'Label'])
        self.dataset['Aggregated_embedding'] = self.dataset['Embedding'].apply(self._aggregate_embeddings)
        self.dataset.to_csv('embedded_dataset.csv')
        self.classify_with_logreg()
        self.plot_important_words()
        self.plot_aopc()
        self.dataset.to_csv('embedded_dataset.csv')

    def _openai_image_description(self, prompt: str) -> Union[str, None]:
        """
        Generate image description with the use of OpenAI API.
        :param prompt: Text specification of the instruction for the LLM.
        :type prompt: str
        :return: Text description of the image if no exception occurs, None otherwise
        """
        self.client = OpenAI(api_key=self.open_ai_key)

        for file, label in self._get_image_files_with_class(self.data_folder):
            encoded_image = self._encode_image(file)
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

                # Extract and yield the description
                yield file, response.choices[0].message.content, self._get_openai_embeddings(
                    response.choices[0].message.content), label
            except Exception as e:
                print(f'Error: {e}')
                yield file, None, np.zeros(1536)

    def _local_image_description(self, prompt: str) -> Union[str, None]:
        """
        Generate image description using the local Hugging Face transformer model.
        :param file: Path to the image file.
        :param prompt: Text prompt for the model.
        :return: Generated description.
        """
        for file, label in self._get_image_files_with_class(self.data_folder):
            try:
                encoded_image = Image.open(file)
                message = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in 7 words. Articles and prepositions count as words. Make sure to use exactly 7 words, be concise."
                            },

                        ]
                    }

                ]
                input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
                inputs = self.processor(
                    encoded_image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.model.device)

                output = self.model.generate(**inputs, max_new_tokens=30)
                decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
                decoded_output = decoded_output.strip()
                decoded_output = decoded_output.split('assistant\n\n')[1]
                print(f">{ decoded_output = }<")
                print(file)
                torch.cuda.empty_cache()
                del output
                yield file, decoded_output, self._get_local_embeddings(decoded_output), label

            except Exception as e:
                    print(f'Error: {e}')
                    return None

    def _get_openai_embeddings(self, sentence: str) -> np.array:
        """
        Generate embeddings for the image's word description using openai.
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
                    model=self.openai_embedding
                )
                embeddings.append(list(response.data[0].embedding))
            except Exception as e:
                print(f"Error generating embedding for '{word}': {e}")

        # If embeddings are successfully generated, return them; otherwise, return zeros
        if embeddings:
            return embeddings
        else:
            return np.zeros(1536)  # ADA embedding size is 1536

    def _get_local_embeddings(self, sentence: str) -> np.array:
        """
        Generate embeddings for the image's word description.
        :param sentence: Word description of an image.
        :type sentence: str
        :return: Embedding of length 1536
        """
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.local_embedding)
        self.embedding_model = AutoModel.from_pretrained(self.local_embedding)

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.embedding_model.to(self.device)
        embeddings = []
        for word in sentence.split():
            inputs = self.bert_tokenizer(word, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to GPU
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)

            # Get the embeddings from the [CLS] token
            embeddings.append(np.squeeze(outputs.last_hidden_state[:, 0, :].cpu().numpy()))  # Move embeddings back to CPU
        return embeddings

    def classify_with_logreg(self):
        """Use logistic regression to classify the instances and get feature weights."""
        # Extract features and labels
        X = np.stack(self.dataset['Aggregated_embedding'].values)
        y = self.dataset['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Logistic Regression classifier
        self.logreg_model = LogisticRegression()
        self.logreg_model.fit(X_train, y_train)

        accuracy = self.logreg_model.score(X_test, y_test)
        print("Accuracy:", accuracy)
        self.dataset['Feature_Importance'] = None  # Initialize column
        self.dataset['Sorted_Words_by_Importance'] = None

        for idx in range(len(self.dataset)):
            description = self.dataset.Description[idx]

            # Get the original prediction probability
            original_embeddings = []
            words = description.split()
            word_indices = [description.split().index(word) for word in words if word in description.split()]
            word_embeddings = [self.dataset['Embedding'].iloc[idx][i] for i in word_indices]

            if word_embeddings:
                original_aggregated_embedding = np.mean(word_embeddings, axis=0)
            else:
                original_aggregated_embedding = np.zeros_like(self.dataset['Embedding'].iloc[idx][0])

            original_embeddings.append(original_aggregated_embedding)
            original_embeddings = np.array(original_embeddings)

            original_prob = self.logreg_model.predict_proba(original_embeddings)[0]
            importance_scores = {}

            for word in words:
                perturbed_text = ' '.join(w for w in words if w != word)
                perturbed_words = perturbed_text.split()
                perturbed_word_indices = [description.split().index(w) for w in perturbed_words if w in description.split()]
                perturbed_word_embeddings = [self.dataset['Embedding'].iloc[idx][i] for i in perturbed_word_indices]

                if perturbed_word_embeddings:
                    perturbed_aggregated_embedding = np.mean(perturbed_word_embeddings, axis=0)
                else:
                    perturbed_aggregated_embedding = np.zeros_like(self.dataset['Embedding'].iloc[idx][0])

                perturbed_embeddings = np.array([perturbed_aggregated_embedding])
                perturbed_prob = self.logreg_model.predict_proba(perturbed_embeddings)[0]
                drop = original_prob - perturbed_prob

                target_class = self.dataset.Label[idx]
                class_index = self.logreg_model.classes_.tolist().index(target_class)
                importance_scores[word] = drop[class_index]

            # Sort words by importance score
            sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            importance_str = ','.join(f"{word}:{score:.4f}" for word, score in sorted_importance)
            self.dataset.at[idx, 'Feature_Importance'] = importance_str

            sorted_words_list = [word for word, score in sorted_importance]
            sorted_words_str = ','.join(sorted_words_list)
            self.dataset.at[idx, 'Sorted_Words_by_Importance'] = sorted_words_str

        self.dataset['Sorted_Words_by_Importance_processed'] = self.dataset['Sorted_Words_by_Importance'].apply(
            self._preprocess_text)

        self.df_AOPC = self.dataset[X_train.shape[0] + 1:].reset_index(drop=True)

    def plot_aopc(self):
        """"""
        # Calculate AOPC by iteratively removing the most important word for each description
        VECTOR_SIZE = 1536
        max_K = 6  # Maximum number of top words to remove
        avg_drops = []

        for K in tqdm(range(1, max_K + 1)):  # Loop through K from 1 to max_K (10)
            drops = []
            for idx, row in self.df_AOPC.iterrows():
                original_text = row['Description']
                word_to_index = {word: i for i, word in enumerate(original_text.split())}

                # Calculate original probability
                original_word_indices = [i for word, i in word_to_index.items()]
                original_embeddings = [np.array(row['Embedding'][i]) for i in original_word_indices]
                original_aggregated_embedding = np.mean(original_embeddings,
                                                        axis=0) if original_embeddings else np.zeros(VECTOR_SIZE)
                original_probs = self.logreg_model.predict_proba([original_aggregated_embedding])[0]
                original_class = np.argmax(original_probs)  # Determine the predicted class
                original_prob = original_probs[original_class]

                # Initialize text for iterative removal
                altered_text = original_text

                # Iteratively remove the most important word
                for _ in range(K):
                    word_importances = {}
                    words = altered_text.split()

                    # Calculate importance of each word
                    for word in words:
                        # Remove the word and calculate the new probability
                        temp_text = ' '.join(w for w in words if w != word)
                        temp_word_indices = [word_to_index[w] for w in temp_text.split() if w in word_to_index]
                        temp_embeddings = [np.array(row['Embedding'][i]) for i in temp_word_indices]
                        temp_aggregated_embedding = np.mean(temp_embeddings, axis=0) if temp_embeddings else np.zeros(
                            VECTOR_SIZE)
                        temp_prob = self.logreg_model.predict_proba([temp_aggregated_embedding])[0][original_class]

                        # Calculate the drop in probability
                        word_importances[word] = original_prob - temp_prob
                    print(word_importances)

                    # Find the most important word (with the highest drop in probability)
                    if word_importances:
                        most_important_word = max(word_importances, key=word_importances.get)

                        # Remove the most important word from the text
                        altered_text = ' '.join(w for w in altered_text.split() if w != most_important_word)

                        # Recalculate the probability after removing the word
                        altered_word_indices = [word_to_index[w] for w in altered_text.split() if w in word_to_index]
                        altered_embeddings = [np.array(row['Embedding'][i]) for i in altered_word_indices]
                        altered_aggregated_embedding = np.mean(altered_embeddings,
                                                               axis=0) if altered_embeddings else np.zeros(VECTOR_SIZE)
                        altered_prob = self.logreg_model.predict_proba([altered_aggregated_embedding])[0][
                            original_class]
                    else:
                        altered_prob = original_prob

                    # Calculate and store the drop for this iteration
                    drop = original_prob - altered_prob
                    drops.append(drop)

            avg_drops.append(np.mean(drops))

        # Plot AOPC results
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_K + 1), avg_drops, marker='o')
        plt.xlabel('Number of Words Removed (K)')
        plt.ylabel('Average Output Probability Change (AOPC)')
        plt.title('AOPC vs. Number of Important Words Removed')
        plt.grid(True)
        plt.show()

    def plot_important_words(self):
        # Initialize a list to store the most important words
        most_important_words = []

        # Extract the most important word from each row
        for idx in range(len(self.dataset)):
            # Extract the first word from the 'Sorted_Words' column
            sorted_words = self.dataset.at[idx, 'Sorted_Words_by_Importance_processed']
            if sorted_words:
                most_important_word = sorted_words.split(',')[0].lower()
                most_important_words.append(most_important_word)

        # Create a DataFrame to count occurrences of each most important word
        importance_word_counts = pd.Series(most_important_words).value_counts().reset_index()
        importance_word_counts.columns = ['Word', 'Count']

        # Get the top 10 most important words
        top_10_words = importance_word_counts.head(10)

        # Plotting the top 10 results
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_10_words, x='Word', y='Count', palette='viridis', hue='Count')
        plt.title('Top 10 Most Important Words Across Images')
        plt.xlabel('Word')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show plot
        plt.show()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess the image descriptions.
        :param text: word description of the image
        :type text: string
        :return: lemmatized, normalized text
        """
        # Split the text on commas
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')

        words = text.split(',')
        lemmatized_words = []
        for word in words:
            # Strip leading/trailing whitespace
            word = word.strip()
            # Remove punctuation attached to words (e.g., periods)
            word = re.sub(r'[^\w\s]', '', word)
            # Proceed only if the word is not empty
            if word:
                doc = nlp(word)
                # Lemmatize the word
                lemma = ' '.join([token.lemma_ for token in doc])
                lemmatized_words.append(lemma)
        # Join the lemmatized words with a single comma, ignoring empty entries
        result = ','.join(lemmatized_words)
        return result

    @staticmethod
    def _encode_image(image_path: str):
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
        return np.mean(np.array(embedding_list), axis=0)

    @staticmethod
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
