"""smer_visual.py"""
from pathlib import Path
from tkinter import Image
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from transformers import MllamaForConditionalGeneration, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from PIL import Image
from typing import Union, Optional

from utils import _get_image_files_with_class, _encode_image, _aggregate_embeddings, _preprocess_text


def image_description(
        model: Union[str, Path],
        data_folder: Union[str, Path],
        api_key: Optional[str] = None,
        user_prompt: str = "Describe this image in 7 words. Be concise, try to maximize the information about the objects in this image.",
) -> dict:
    """
        Generate image descriptions using either OpenAI or local models.

        Args:
            model (Union[str, Path]): Model identifier (e.g., 'gpt-4o-mini', '4o', 'o3') or path to local model.
            data_folder (Union[str, Path]): Path to folder containing image files in class subfolders.
            api_key (Optional[str]): OpenAI API key (required for OpenAI models).
            user_prompt (str): Custom prompt for image description generation.

        Returns:
            dict: Dictionary mapping file paths to their descriptions, labels, and errors.

        Example:
            >>> from smer_visual.smer_visual import image_description
            >>> descriptions = image_description(
            ...     model="gpt-4o-mini",
            ...     data_folder="images/",
            ...     api_key="your_api_key",
            ...     user_prompt="Describe this image in 7 words."
            ... )
            >>> print(descriptions)
            {'image1.jpg': {'label': 'cat', 'description': 'A cat sitting on a mat', 'error': None},
             'image2.jpg': {'label': 'dog', 'description': 'A dog playing with a ball', 'error': None}}
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
            local_model = MllamaForConditionalGeneration.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

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
                    ).to(local_model.device)

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


def get_description_embeddings(
        descriptions: dict,
        embedding_model: Union[str, Path],
        api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
        Generate embeddings for image descriptions using either OpenAI or local models.

        Args:
            descriptions (dict): Dictionary output from `image_description()`.
            embedding_model (Union[str, Path]): Model identifier for OpenAI or path to local model.
            api_key (Optional[str]): OpenAI API key (required for OpenAI embeddings).

        Returns:
            pd.DataFrame: Pandas DataFrame with columns: image, description, embedding, label.

        Example:
            >>> from smer_visual.smer_visual import get_description_embeddings
            >>> descriptions = {
            ...     "image1.jpg": {"description": "A cat sitting on a mat", "label": "cat"},
            ...     "image2.jpg": {"description": "A dog playing with a ball", "label": "dog"}
            ... }
            >>> embeddings_df = get_description_embeddings(
            ...     descriptions=descriptions,
            ...     embedding_model="text-embedding-ada-002",
            ...     api_key="your_api_key"
            ... )
            >>> print(embeddings_df)
                   image               description                                           embedding label
            0  image1.jpg  A cat sitting on a mat  [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...]]   cat
            1  image2.jpg  A dog playing with a ball  [[0.7, 0.8, 0.9, ...], [0.1, 0.2, 0.3, ...]]   dog
        """
    OPENAI_MODELS = {'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'}
    results = descriptions.copy()

    def process_with_openai(client: OpenAI) -> None:
        """
        Generate embeddings for each word in the description using OpenAI API.
        """
        for file_path in results:
            if results[file_path]['description']:
                try:
                    sentence = results[file_path]['description']
                    embeddings = []

                    # Generate embeddings for each word in the sentence
                    for word in sentence.split():
                        response = client.embeddings.create(
                            input=word,
                            model=embedding_model
                        )
                        word_embedding = response.data[0].embedding
                        embeddings.append(word_embedding)

                    results[file_path]['embedding'] = embeddings
                except Exception as e:
                    results[file_path]['embedding'] = None
                    results[file_path]['error'] = f"Embedding error: {str(e)}"
            else:
                results[file_path]['embedding'] = None

    def process_with_local_model() -> None:
        """
        Generate embeddings for each word in the description using a local model.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            model = AutoModel.from_pretrained(embedding_model)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            for file_path in results:
                if results[file_path]['description']:
                    try:
                        sentence = results[file_path]['description']
                        embeddings = []

                        # Generate embeddings for each word in the sentence
                        for word in sentence.split():
                            inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True, max_length=512)
                            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
                            with torch.no_grad():
                                outputs = model(**inputs)
                            # Get the embeddings from the [CLS] token
                            embeddings.append(np.squeeze(outputs.last_hidden_state[:, 0, :].cpu().numpy()))  # Move embeddings back to CPU

                        results[file_path]['embedding'] = embeddings
                    except Exception as e:
                        results[file_path]['embedding'] = None
                        results[file_path]['error'] = f"Embedding error: {str(e)}"
                else:
                    results[file_path]['embedding'] = None

        except Exception as e:
            raise ValueError(f"Error initializing local model: {str(e)}")

    if isinstance(embedding_model, str) and embedding_model in OPENAI_MODELS:
        if not api_key:
            raise ValueError("API key required for OpenAI embeddings")
        process_with_openai(OpenAI(api_key=api_key))
    else:
        process_with_local_model()

    data = []
    for file_path, values in results.items():
        data.append({
            "image": file_path,
            "description": values.get("description"),
            "embedding": values.get("embedding"),
            "label": values.get("label")
        })

    return pd.DataFrame(data)


def aggregate_embeddings(embedding_list):
    """Flatten the list of lists and take the mean along the axis"""
    return np.mean(np.array(embedding_list), axis=0)


def classify_with_logreg(dataset: pd.DataFrame, X_train,
                         logreg_model: LogisticRegression) -> (pd.DataFrame, pd.DataFrame):
    """
        Use logistic regression to classify the instances and get feature weights.

        Args:
            dataset (pd.DataFrame): DataFrame containing descriptions, embeddings, and labels.
            X_train (np.ndarray): Training data embeddings.
            logreg_model (LogisticRegression): Pre-trained logistic regression model.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: AOPC DataFrame.
                - pd.DataFrame: Updated dataset with feature importance and sorted words.

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> from smer_visual.smer_visual import classify_with_logreg
            >>> dataset = pd.DataFrame({
            ...     "description": ["A cat sitting on a mat", "A dog playing with a ball"],
            ...     "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            ...     "label": ["cat", "dog"]
            ... })
            >>> X_train = np.random.rand(2, 3)
            >>> logreg_model = LogisticRegression()
            >>> aopc_df, updated_dataset = classify_with_logreg(dataset, X_train, logreg_model)
            >>> print(aopc_df)
            >>> print(updated_dataset)
        """
    dataset['feature_importance'] = None  # Initialize column
    dataset['sorted_words_by_importance'] = None

    for idx in range(len(dataset)):
        description = dataset.description[idx]

        # Get the original prediction probability
        original_embeddings = []
        words = description.split()
        word_indices = [description.split().index(word) for word in words if word in description.split()]
        word_embeddings = [dataset['embedding'].iloc[idx][i] for i in word_indices]

        if word_embeddings:
            original_aggregated_embedding = np.mean(word_embeddings, axis=0)
        else:
            original_aggregated_embedding = np.zeros_like(dataset['embedding'].iloc[idx][0])

        original_embeddings.append(original_aggregated_embedding)
        original_embeddings = np.array(original_embeddings)

        original_prob = logreg_model.predict_proba(original_embeddings)[0]
        importance_scores = {}

        for word in words:
            perturbed_text = ' '.join(w for w in words if w != word)
            perturbed_words = perturbed_text.split()
            perturbed_word_indices = [description.split().index(w) for w in perturbed_words if
                                      w in description.split()]
            perturbed_word_embeddings = [dataset['embedding'].iloc[idx][i] for i in perturbed_word_indices]

            if perturbed_word_embeddings:
                perturbed_aggregated_embedding = np.mean(perturbed_word_embeddings, axis=0)
            else:
                perturbed_aggregated_embedding = np.zeros_like(dataset['embedding'].iloc[idx][0])

            perturbed_embeddings = np.array([perturbed_aggregated_embedding])
            perturbed_prob = logreg_model.predict_proba(perturbed_embeddings)[0]
            drop = original_prob - perturbed_prob

            target_class = dataset.label[idx]
            class_index = logreg_model.classes_.tolist().index(target_class)
            importance_scores[word] = drop[class_index]

        # Sort words by importance score
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        importance_str = ','.join(f"{word}:{score:.4f}" for word, score in sorted_importance)
        dataset.at[idx, 'feature_importance'] = importance_str

        sorted_words_list = [word for word, score in sorted_importance]
        sorted_words_str = ','.join(sorted_words_list)
        dataset.at[idx, 'sorted_words_by_importance'] = sorted_words_str

    dataset['sorted_words_by_importance_processed'] = dataset['sorted_words_by_importance'].apply(
        _preprocess_text)

    df_aopc = dataset[X_train.shape[0] + 1:].reset_index(drop=True)
    return df_aopc, dataset


def compute_aopc(df, top_words, max_k, logreg_model):
    """
      Plot AOPC curves for SMER and LIME importance scores.

      Args:
          df_aopc (pd.DataFrame): DataFrame containing descriptions and embeddings.
          logreg_model (LogisticRegression): Pre-trained logistic regression model.
          max_k (int): Maximum number of words to remove.

      Returns:
          None: Displays the plot.

      Example:
          >>> from smer_visual.smer_visual import plot_aopc
          >>> plot_aopc(df_aopc, logreg_model, max_k=6)
      """
    avg_drops = []
    for K in range(0, max_k + 1):
        drops = []
        for idx2, row2 in df.iterrows():
            text2 = row2['description']
            original_probs2 = _predict_proba_for_text(text2, row2, logreg_model)
            original_class2 = np.argmax(original_probs2)
            original_prob2 = original_probs2[original_class2]

            # Which of the top_words appear in the text?
            words_in_text2 = text2.split()
            top_in_text2 = [w for w in top_words if w in words_in_text2]

            # Remove up to K
            words_to_remove2 = top_in_text2[:K]
            if not words_to_remove2:
                drop2 = 0.0
            else:
                altered_text2 = ' '.join(w for w in words_in_text2 if w not in words_to_remove2)
                altered_probs2 = _predict_proba_for_text(altered_text2, row2, logreg_model)
                altered_prob2 = altered_probs2[original_class2]
                drop2 = original_prob2 - altered_prob2

            drops.append(drop2)

        avg_drops.append(np.mean(drops) if drops else 0.0)

    return avg_drops


def build_custom_predict(row, logreg_model):
    """
    Returns a function that LIME calls like classifier_fn(texts:list[str]) -> np.ndarray
    for the old aggregator approach.
    """
    def predict_for_lime(texts):
        emb_list = []
        original_tokens = row['description'].split()
        word_to_index = {w: i for i, w in enumerate(original_tokens)}

        for t in texts:
            t_words = t.split()
            valid_indices = [word_to_index[x] for x in t_words if x in word_to_index]
            if len(valid_indices) > 0:
                emb = np.mean([np.array(row['embedding'][ix]) for ix in valid_indices], axis=0)
            else:
                emb = np.zeros(len(row['embedding'][0]))
            emb_list.append(emb)
        return logreg_model.predict_proba(emb_list)
    return predict_for_lime


def plot_aopc(df_aopc, logreg_model, max_k=6):
    """
    Plot AOPC curves for SMER and LIME importance scores.

    Args:
        df_aopc (pd.DataFrame): DataFrame containing descriptions and embeddings.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.
        max_k (int): Maximum number of words to remove.

    Returns:
        None: Displays the plot.

    Example:
        >>> from smer_visual.smer_visual import plot_aopc
        >>> plot_aopc(df_aopc, logreg_model, max_k=6)
    """
    # SMER importance score calculation
    smer_rows = []
    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing SMER importances"):
        text = row['description']
        original_probs = _predict_proba_for_text(text, row, logreg_model)
        original_class = np.argmax(original_probs)
        original_prob = original_probs[original_class]

        words = text.split()
        for w in words:
            # Remove this word
            altered_text = ' '.join(token for token in words if token != w)
            if not altered_text.strip():
                drop = 0.0
            else:
                altered_probs = _predict_proba_for_text(altered_text, row, logreg_model)
                drop = original_prob - altered_probs[original_class]

            smer_rows.append({
                'word': w,
                'importance': abs(drop)
            })

    smer_df = pd.DataFrame(smer_rows)
    if 'label' in df_aopc.columns:
        class_names = df_aopc['label'].unique().tolist()
    else:
        class_names = []

    # Lime imporance score calculation
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    lime_rows = []

    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing LIME importances"):
        text = row['description']
        lime_predict_fn = build_custom_predict(row, logreg_model)

        # Explain instance
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=lime_predict_fn,
            num_features=len(text.split())
        )

        local_importances = exp.as_list()
        for w, imp in local_importances:
            lime_rows.append({
                'word': w,
                'importance': abs(imp)
            })

    lime_df = pd.DataFrame(lime_rows)

    # Result aggregation, selection of top words

    # SMER top 20
    global_importances_smer = (
        smer_df.groupby('word')['importance'].mean()
        .reset_index()
        .sort_values('importance', ascending=False)
    )
    top_words_smer = global_importances_smer['word'].head(20).tolist()

    # LIME top 20
    global_importances_lime = (
        lime_df.groupby('word')['importance'].mean()
        .reset_index()
        .sort_values('importance', ascending=False)
    )
    top_words_lime = global_importances_lime['word'].head(20).tolist()

    # Compute AOPC score

    AOPC_SMER = compute_aopc(df_aopc, top_words_smer, max_k, logreg_model)
    AOPC_LIME = compute_aopc(df_aopc, top_words_lime, max_k, logreg_model)

    # Visualize results

    plt.figure(figsize=(10, 6))
    x_values = range(0, max_k + 1)
    plt.plot(x_values, AOPC_SMER, marker='o', label='SMER')
    plt.plot(x_values, AOPC_LIME, marker='x', label='LIME')

    plt.xlabel('Number of Words Removed (K)')
    plt.ylabel('Average Probability Drop')
    plt.title('Comparison of AOPC vs. Number of Important Words Removed')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_important_words(dataset):
    """
     Plot the top 10 most important words across images.

     Args:
         dataset (pd.DataFrame): DataFrame containing sorted words by importance.

     Returns:
         None: Displays the plot.

     Example:
         >>> from smer_visual.smer_visual import plot_important_words
         >>> plot_important_words(dataset)
    """
    most_important_words = []

    for idx in range(len(dataset)):
        sorted_words = dataset.at[idx, 'sorted_words_by_importance_processed']
        if sorted_words:
            most_important_word = sorted_words.split(',')[0].lower()
            most_important_words.append(most_important_word)

    importance_word_counts = pd.Series(most_important_words).value_counts().reset_index()
    importance_word_counts.columns = ['Word', 'Count']

    top_10_words = importance_word_counts.head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_10_words, x='Word', y='Count', palette='viridis', hue='Count')
    plt.title('Top 10 Most Important Words Across Images')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def _predict_proba_for_text(text, row, logreg_model):
    """
    Predict probabilities for a given text using logistic regression.

    Args:
        text (str): Text description.
        row (pd.Series): Row containing embeddings and labels.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.

    Returns:
        np.ndarray: Predicted probabilities.

    Example:
        >>> from smer_visual.smer_visual import _predict_proba_for_text
        >>> text = "A cat sitting on a mat"
        >>> row = dataset.iloc[0]
        >>> probs = _predict_proba_for_text(text, row, logreg_model)
        >>> print(probs)
    """
    original_tokens = row['description'].split()
    word_to_index = {w: i for i, w in enumerate(original_tokens)}

    text_words = text.split()
    valid_indices = [word_to_index[w] for w in text_words if w in word_to_index]

    if len(valid_indices) > 0:
        emb = np.mean([np.array(row['embedding'][ix]) for ix in valid_indices], axis=0)
    else:
        emb = np.zeros(len(row['embedding'][0]))

    probs = logreg_model.predict_proba([emb])[0]
    return probs
