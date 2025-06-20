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


def get_description_embeddings(
        descriptions: dict,
        embedding_model: Union[str, Path],
        api_key: Optional[str] = None,
) -> dict:
    """
    Generate embeddings for image descriptions using either OpenAI or local models.

    Args:
        descriptions: Dictionary output from image_description()
        embedding_model: Model identifier for OpenAI or path to local model
        api_key: OpenAI API key (required for OpenAI embeddings)

    Returns:
        Dictionary with the same structure as input, plus embeddings:
        - 'embedding': Non-aggregated embeddings
        - 'aggregated_embedding': Aggregated embeddings
    """
    OPENAI_MODELS = {'text-embedding-ada-002'}
    results = descriptions.copy()

    def process_with_openai(client: OpenAI) -> None:
        for file_path in results:
            if results[file_path]['description']:
                try:
                    response = client.embeddings.create(
                        input=results[file_path]['description'],
                        model=embedding_model
                    )
                    embeddings = response.data[0].embedding
                    results[file_path]['embedding'] = embeddings
                    results[file_path]['aggregated_embedding'] = _aggregate_embeddings([embeddings])
                except Exception as e:
                    results[file_path]['embedding'] = None
                    results[file_path]['aggregated_embedding'] = None
                    results[file_path]['error'] = f"Embedding error: {str(e)}"
            else:
                results[file_path]['embedding'] = None
                results[file_path]['aggregated_embedding'] = None

    def process_with_local_model() -> None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            model = AutoModel.from_pretrained(embedding_model)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            for file_path in results:
                if results[file_path]['description']:
                    try:
                        inputs = tokenizer(
                            results[file_path]['description'],
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model(**inputs)

                        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        results[file_path]['embedding'] = embeddings
                        results[file_path]['aggregated_embedding'] = _aggregate_embeddings([embeddings])
                    except Exception as e:
                        results[file_path]['embedding'] = None
                        results[file_path]['aggregated_embedding'] = None
                        results[file_path]['error'] = f"Embedding error: {str(e)}"
                else:
                    results[file_path]['embedding'] = None
                    results[file_path]['aggregated_embedding'] = None

        except Exception as e:
            raise ValueError(f"Error initializing local model: {str(e)}")

    if isinstance(embedding_model, str) and embedding_model in OPENAI_MODELS:
        if not api_key:
            raise ValueError("API key required for OpenAI embeddings")
        process_with_openai(OpenAI(api_key=api_key))
    else:
        process_with_local_model()

    return results


def classify_with_logreg(dataset: pd.DataFrame, X_train,
                         logreg_model: LogisticRegression) -> (pd.DataFrame, pd.DataFrame):
    """
    Use logistic regression to classify the instances and get feature weights.
    Maintains the original logic, focusing on improved readability.
    """
    dataset['Feature_Importance'] = None
    dataset['Sorted_Words_by_Importance'] = None

    # Compute importance of each word in the description
    for idx in range(len(dataset)):
        description = dataset.Description[idx]
        words = description.split()

        # Compute the aggregated embedding for the full description
        word_embeddings = [dataset['Embedding'].iloc[idx][i] for i in range(len(words)) if words[i] in words]
        if word_embeddings:
            original_agg_emb = np.mean(word_embeddings, axis=0)
        else:
            original_agg_emb = np.zeros_like(dataset['Embedding'].iloc[idx][0])

        # Predict probability with all words
        original_prob = logreg_model.predict_proba(np.array([original_agg_emb]))[0]

        # Calculate effectiveness of each word
        importance_scores = {}
        for word in words:
            perturbed_text = ' '.join(w for w in words if w != word)
            perturbed_words = perturbed_text.split()
            perturbed_embeddings = [
                dataset['Embedding'].iloc[idx][i]
                for i in range(len(perturbed_words))
                if perturbed_words[i] in perturbed_words
            ]

            if perturbed_embeddings:
                perturbed_agg_emb = np.mean(perturbed_embeddings, axis=0)
            else:
                perturbed_agg_emb = np.zeros_like(dataset['Embedding'].iloc[idx][0])

            perturbed_prob = logreg_model.predict_proba(np.array([perturbed_agg_emb]))[0]
            drop = original_prob - perturbed_prob
            class_index = logreg_model.classes_.tolist().index(dataset.Label[idx])
            importance_scores[word] = drop[class_index]

        # Store sorted importance
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        dataset.at[idx, 'Feature_Importance'] = ','.join(f"{w}: {s:.4f}" for w, s in sorted_importance)
        dataset.at[idx, 'Sorted_Words_by_Importance'] = ','.join([w for w, _ in sorted_importance])

    # Preprocess sorted words for the specified range
    dataset['Sorted_Words_by_Importance_processed'] = dataset['Sorted_Words_by_Importance'].apply(_preprocess_text)

    # Split final outputs as in the original code
    df_aopc = dataset[X_train.shape[0] + 1:].reset_index(drop=True)
    return df_aopc, dataset


def compute_aopc(df, top_words, max_k, logreg_model):
    """
    Remove up to K of the specified `top_words` from each text
    and measure probability drop of the original predicted class.
    """
    avg_drops = []
    for K in range(0, max_k + 1):
        drops = []
        for idx2, row2 in df.iterrows():
            text2 = row2['Description']
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
        original_tokens = row['Description'].split()
        word_to_index = {w: i for i, w in enumerate(original_tokens)}

        for t in texts:
            t_words = t.split()
            valid_indices = [word_to_index[x] for x in t_words if x in word_to_index]
            if len(valid_indices) > 0:
                emb = np.mean([np.array(row['Embedding'][ix]) for ix in valid_indices], axis=0)
            else:
                emb = np.zeros(len(row['Embedding'][0]))
            emb_list.append(emb)
        return logreg_model.predict_proba(emb_list)
    return predict_for_lime


def plot_aopc(df_aopc, logreg_model, max_k=6):
    """
    A single function that:
      1) Computes SMER importances for each row in df_AOPC.
      2) Computes LIME importances for each row in df_AOPC.
      3) Aggregates them globally to find top 20 words.
      4) Computes AOPC by removing up to K of these words from each text.
      5) Plots the resulting AOPC curves for SMER and LIME.

    """
    # SMER importance score calculation
    smer_rows = []
    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing SMER importances"):
        text = row['Description']
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
    if 'Label' in df_aopc.columns:
        class_names = df_aopc['Label'].unique().tolist()
    else:
        class_names = []

    # Lime imporance score calculation
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    lime_rows = []

    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing LIME importances"):
        text = row['Description']
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
    """"""
    # Initialize a list to store the most important words
    most_important_words = []

    # Extract the most important word from each row
    for idx in range(len(dataset)):
        # Extract the first word from the 'Sorted_Words' column
        sorted_words = dataset.at[idx, 'Sorted_Words_by_Importance_processed']
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

def _predict_proba_for_text(text, row, logreg_model):
    """
    Reconstructs embeddings from row['Embedding'] for the words present in `text`,
    then uses mean pooling + logreg_model.predict_proba().
    """
    original_tokens = row['Description'].split()
    word_to_index = {w: i for i, w in enumerate(original_tokens)}

    text_words = text.split()
    valid_indices = [word_to_index[w] for w in text_words if w in word_to_index]

    if len(valid_indices) > 0:
        emb = np.mean([np.array(row['Embedding'][ix]) for ix in valid_indices], axis=0)
    else:
        emb = np.zeros(len(row['Embedding'][0]))

    probs = logreg_model.predict_proba([emb])[0]
    return probs
