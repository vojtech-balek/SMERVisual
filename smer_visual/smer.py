"""smer.py"""
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
from typing import Union, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageFont
from .utils import _get_image_files_with_class, _encode_image, _preprocess_text


_local_model_cache = {}

def image_descriptions(
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
            >>> import os
            >>> from dotenv import load_dotenv
            >>> load_dotenv()
            >>> descriptions = image_description(
            ...     model="gpt-4o-mini",
            ...     data_folder="images/",
            ...     api_key=os.getenv("OPENAI_API_KEY"),
            ...     user_prompt="Describe this image in 7 words. Be concise, try to maximize the information about the objects in this image."
            ... )
            >>> print(descriptions)
            {'image1.jpg': {'label': 'cat', 'description': 'A cat sitting on a mat', 'error': None},
             'image2.jpg': {'label': 'dog', 'description': 'A dog playing with a ball', 'error': None}}
        """
    OPENAI_MODELS = {'gpt-4o-mini', '4o', 'o3', 'o3-mini', 'o3-pro',
                     'o1-pro', 'gpt-4.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano'}

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
        """
        Generate image descriptions using a local model.
        Uses a cache to avoid reloading the model for repeated calls.
        """
        results = {}
        cache_key = str(model)  # model is from outer scope

        try:
            if cache_key not in _local_model_cache:
                processor = AutoProcessor.from_pretrained(model)
                tokenizer = AutoTokenizer.from_pretrained(model)
                local_model = MllamaForConditionalGeneration.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                _local_model_cache[cache_key] = {
                    'processor': processor,
                    'tokenizer': tokenizer,
                    'model': local_model
                }

            cached = _local_model_cache[cache_key]
            processor = cached['processor']
            tokenizer = cached['tokenizer']
            local_model = cached['model']

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


def embed_descriptions(
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
            >>> from smer_visual.smer_visual import embed_descriptions
            >>> import os
            >>> from dotenv import load_dotenv
            >>> descriptions = {
            ...     "image1.jpg": {"description": "A cat sitting on a mat", "label": "cat"},
            ...     "image2.jpg": {"description": "A dog playing with a ball", "label": "dog"}
            ... }
            >>> load_dotenv()
            >>> embeddings_df = embed_descriptions(
            ...     descriptions=descriptions,
            ...     embedding_model="text-embedding-ada-002",
            ...     api_key=os.getenv("OPENAI_API_KEY")
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
        Generate embeddings for words in batches using OpenAI API.
        """
        BATCH_SIZE = 100  # Adjust based on API limits and needs

        for file_path in results:
            if results[file_path]['description']:
                try:
                    sentence = results[file_path]['description']
                    words = sentence.split()
                    embeddings = []

                    # Process words in batches
                    for i in range(0, len(words), BATCH_SIZE):
                        batch = words[i:i + BATCH_SIZE]
                        response = client.embeddings.create(
                            input=batch,
                            model=embedding_model
                        )
                        # Extract embeddings in the same order as input
                        batch_embeddings = [data.embedding for data in response.data]
                        embeddings.extend(batch_embeddings)

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
    return filter_missing_embeddings(pd.DataFrame(data))


def filter_missing_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where embeddings are missing.

    Args:
        df: DataFrame from embed_descriptions with potential missing embeddings.

    Returns:
        DataFrame with only valid embeddings.
    """
    valid_mask = df['embedding'].apply(lambda x: x is not None and len(x) > 0)
    filtered_df = df[valid_mask].copy()
    removed = len(df) - len(filtered_df)

    if removed > 0:
        print(f"Removed {removed} rows with missing embeddings.")

    return filtered_df


def aggregate_embeddings(embedding_list):
    """Flatten the list of lists and take the mean along the axis"""
    return np.mean(np.array(embedding_list), axis=0)


def _predict_proba_for_text(text, row, logreg_model):
    """
    Predict probabilities for a given text using logistic regression.

    This mimics sklearn's LogisticRegression.predict_proba:
      - For multinomial classifiers (n_classes > 2) apply softmax to logits.
      - For binary classifiers return [1 - p, p] where p is the sigmoid of the logit.

    Args:
        text (str): Text description.
        row (pd.Series): Row containing embeddings and labels.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.

    Returns:
        np.ndarray: Predicted probabilities (shape = n_classes, sums to 1).
    """
    original_tokens = row['description'].split()
    word_to_index = {w: i for i, w in enumerate(original_tokens)}

    text_words = text.split()
    valid_indices = [word_to_index[w] for w in text_words if w in word_to_index]

    if len(valid_indices) > 0:
        emb = np.mean([np.array(row['embedding'][ix]) for ix in valid_indices], axis=0)
    else:
        emb = np.zeros(len(row['embedding'][0]))

    # Compute logits: shape could be (n_classes,) for multinomial or (1,) for binary (sklearn stores coef_ as (1, n_features))
    logits = np.dot(emb, logreg_model.coef_.T) + logreg_model.intercept_
    logits = np.asarray(logits).ravel()

    # Helper: numerically-stable softmax
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_max = np.max(x)
        ex = np.exp(x - x_max)
        return ex / ex.sum()

    # If multinomial (more than 1 row in coef_), apply softmax
    if getattr(logreg_model, "coef_", None) is not None and logreg_model.coef_.shape[0] > 1:
        probs = _softmax(logits)
    else:
        # Binary case: logits is single value -> sigmoid
        # Ensure scalar
        logit_val = float(logits[0]) if logits.size > 0 else 0.0
        p = 1.0 / (1.0 + np.exp(-logit_val))
        probs = np.array([1.0 - p, p])

    return probs

def classify_lr(dataset: pd.DataFrame, X,
                logreg_model: LogisticRegression) -> (pd.DataFrame, pd.DataFrame):
    """
        Use logistic regression to classify the instances and get feature weights.

        Args:
            dataset (pd.DataFrame): DataFrame containing descriptions, embeddings, and labels.
            X (np.ndarray): Data embeddings.
            logreg_model (LogisticRegression): Pre-trained logistic regression model.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: AOPC DataFrame.
                - pd.DataFrame: Updated dataset with feature importance and sorted words.

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> from smer_visual.smer_visual import classify_lr
            >>> dataset = pd.DataFrame({
            ...     "description": ["A cat sitting on a mat", "A dog playing with a ball"],
            ...     "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            ...     "label": ["cat", "dog"]
            ... })
            >>> X_train = np.random.rand(2, 3)
            >>> logreg_model = LogisticRegression()
            >>> aopc_df, updated_dataset = classify_lr(dataset, X_train, logreg_model)
            >>> print(aopc_df)
            >>> print(updated_dataset)
        """
    dataset['feature_importance'] = None
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

    df_aopc = dataset[X.shape[0] + 1:].reset_index(drop=True)
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
         dataset (pd.DataFrame): DataFrame with top 10 important words and their counts.

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
    importance_word_counts.columns = ['word', 'count']

    top_10_words = importance_word_counts.head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_10_words, x='word', y='count', palette='viridis', hue='count')
    plt.title('Top 10 Most Important Words Across Images')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
    return top_10_words

def _bounding_boxes(image_path: str,
                    df_top_words: pd.DataFrame,
                    model_id: str,
                    device = 'cuda' if torch.cuda.is_available() else 'cpu',
                    box_threshold=0.4,
                    text_threshold=0.3):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    image = Image.open(image_path)
    text = ". ".join(df_top_words['word'].tolist())
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    # Extract the result for the first image
    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    # Create a copy of the image for drawing
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw boxes and labels on the image
    for box, score, label in zip(boxes, scores, labels):
        box = box.tolist()
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        label_text = f"{label}: {score:.2f}"
        text_size = draw.textbbox((0, 0), label_text, font=font)[2:]
        draw.rectangle([(x1, y1 - text_size[1]), (x1 + text_size[0], y1)], fill="red")

        # Draw label text
        draw.text((x1, y1 - text_size[1]), label_text, fill="white", font=font)
    return image_with_boxes


def save_bounding_box_images(
        input_path: Union[str, Path],
        output_folder: Union[str, Path],
        df_top_words: pd.DataFrame,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        box_threshold: float = 0.4,
        text_threshold: float = 0.3
) -> dict:
    """
    Process images with bounding boxes and save to output folder.

    Args:
        input_path: Path to image file or directory with images
        output_folder: Directory where annotated images will be saved
        df_top_words: DataFrame containing important words to detect
        model_id: Model for zero-shot object detection
        device: Device to run model on ('cuda' or 'cpu')
        box_threshold: Threshold for box detection
        text_threshold: Threshold for text detection
    """
    # Convert paths to Path objects
    input_path = Path(input_path)
    output_folder = Path(output_folder)

    # Create output directory if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)
    valid_exts = {".jpg", ".jpeg", ".png"}
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]

    results = {}
    for img_path in image_paths:
        try:
            # Process image with bounding boxes
            image_with_boxes = _bounding_boxes(
                image_path=str(img_path),
                df_top_words=df_top_words,
                model_id=model_id,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            # Create output filename
            output_path = output_folder / f"{img_path.stem}_annotated{img_path.suffix}"

            # Save the annotated image
            image_with_boxes.save(output_path)

            results[str(img_path)] = str(output_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[str(img_path)] = None

    print(f"Completed! Processed {len(results)} images.")