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
from typing import Union, Optional, List, Dict, Any, Tuple
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageFont
from .utils import _get_image_files_with_class, _encode_image, _preprocess_text

# Import LLM Feature Gen package for interpretable feature extraction
try:
    import llm_feature_gen as lfg
    # If the package doesn't expose the functions directly, try to import them from submodules
    # This handles the case where __init__.py is empty (e.g. some installations)
    if not hasattr(lfg, 'discover_features_from_images'):
        try:
            from llm_feature_gen import discover
            from llm_feature_gen import generate

            # Monkey patch functions onto the lfg module alias
            lfg.discover_features_from_images = discover.discover_features_from_images
            lfg.discover_features_from_videos = discover.discover_features_from_videos
            lfg.discover_features_from_texts = discover.discover_features_from_texts
            lfg.discover_features_from_tabular = discover.discover_features_from_tabular
            lfg.generate_features_from_images = generate.generate_features_from_images
            lfg.generate_features_from_videos = generate.generate_features_from_videos
            lfg.generate_features_from_texts = generate.generate_features_from_texts
            lfg.generate_features_from_tabular = generate.generate_features_from_tabular
        except ImportError:
            pass  # If submodules fail, we can't fix it

    # Ensure LocalProvider is available on the module (for older installations or incomplete __init__)
    if not hasattr(lfg, 'LocalProvider'):
        try:
            from llm_feature_gen.providers.local_provider import LocalProvider
            lfg.LocalProvider = LocalProvider
        except ImportError:
            pass

    # Validate that the required API surface is present
    required_attrs = ['discover_features_from_images', 'generate_features_from_images']
    if all(hasattr(lfg, attr) for attr in required_attrs):
        LFG_AVAILABLE = True
    else:
        LFG_AVAILABLE = False

except ImportError:
    lfg = None
    LFG_AVAILABLE = False


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
            >>> descriptions = image_description(
            ...     model="gpt-4o-mini",
            ...     data_folder="images/",
            ...     api_key="your_api_key",
            ...     user_prompt="Describe this image in 7 words. Be concise, try to maximize the information about the objects in this image."
            ... )
            >>> print(descriptions)
            {'image1.jpg': {'label': 'cat', 'description': 'A cat sitting on a mat', 'error': None},
             'image2.jpg': {'label': 'dog', 'description': 'A dog playing with a ball', 'error': None}}
        """
    OPENAI_MODELS = {'gpt-4o-mini', '4o', 'o3', 'o3-mini', 'o3-pro', 'o1-pro', '4.1'}

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
            >>> descriptions = {
            ...     "image1.jpg": {"description": "A cat sitting on a mat", "label": "cat"},
            ...     "image2.jpg": {"description": "A dog playing with a ball", "label": "dog"}
            ... }
            >>> embeddings_df = embed_descriptions(
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

def classify_lr(dataset: pd.DataFrame, X_train,
                logreg_model: LogisticRegression, test_idx=None) -> (pd.DataFrame, pd.DataFrame):
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

    if test_idx is not None:
        df_aopc = dataset.iloc[test_idx].reset_index(drop=True)
    else:
        df_aopc = dataset[X_train.shape[0]:].reset_index(drop=True)
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

    Returns:
        dict: Mapping of original image paths to saved output paths
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

    return results


# =============================================================================
# LLM Feature Gen (lfg) Integration
# =============================================================================

def _check_lfg_available():
    """Check if LLM Feature Gen package is available."""
    if not LFG_AVAILABLE:
        raise ImportError(
            "llm_feature_gen package is not installed. "
            "Please install it to use this functionality: pip install llm-feature-gen"
        )


def discover_features(
        data_source: Union[str, Path, List[str]],
        source_type: str = "images",
        provider: Optional[Any] = None,
        output_dir: Union[str, Path] = "outputs",
        output_filename: Optional[str] = None,
        num_frames: int = 5,
        use_audio: bool = True,
        max_videos_to_sample: int = 5,
        text_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Discover interpretable features from images or videos using LLM Feature Gen.

    This function analyzes a sample of images or videos and proposes a set of
    interpretable features that can be used for classification.

    Args:
        data_source (Union[str, Path, List[str]]): Path to folder containing images/videos,
            or a list of file paths.
        source_type (str): Type of source data - "images", "videos", "texts", or "tabular". Default: "images".
        provider (Optional[Any]): LFG OpenAIProvider instance. If None, creates one from env vars.
        output_dir (Union[str, Path]): Directory to save discovered features JSON. Default: "outputs".
        output_filename (Optional[str]): Custom output filename. Default: "discovered_features.json".
        num_frames (int): Number of frames to extract per video (videos only). Default: 5.
        use_audio (bool): Whether to use audio transcription (videos only). Default: True.
        max_videos_to_sample (int): Maximum videos to sample for discovery. Default: 5.
        text_column (Optional[str]): Column name containing text data (tabular only).

    Returns:
        Dict[str, Any]: Dictionary containing proposed features with their descriptions.

    Example:
        >>> from smer_visual.smer import discover_features
        >>> # Discover features from images
        >>> features = discover_features(
        ...     data_source="images/sample/",
        ...     source_type="images"
        ... )
        >>> print(features)
        {'proposed_features': [{'feature': 'color', 'description': '...'}, ...]}

        >>> # Discover features from videos
        >>> features = discover_features(
        ...     data_source="videos/sample/",
        ...     source_type="videos",
        ...     use_audio=True
        ... )
    """
    _check_lfg_available()

    if source_type == "images":
        return lfg.discover_features_from_images(
            image_paths_or_folder=data_source,
            provider=provider,
            as_set=True,
            output_dir=output_dir,
            output_filename=output_filename,
        )
    elif source_type == "videos":
        videos_input = data_source if isinstance(data_source, list) else str(data_source)
        return lfg.discover_features_from_videos(
            videos_or_folder=videos_input,
            provider=provider,
            num_frames=num_frames,
            output_dir=output_dir,
            output_filename=output_filename,
            use_audio=use_audio,
            max_videos_to_sample=max_videos_to_sample,
        )
    elif source_type == "texts":
        return lfg.discover_features_from_texts(
            texts_or_file=data_source,
            provider=provider,
            as_set=True,
            output_dir=output_dir,
            output_filename=output_filename,
        )
    elif source_type == "tabular":
        if text_column is None:
            raise ValueError("text_column is required when source_type='tabular'")
        return lfg.discover_features_from_tabular(
            file_or_folder=data_source,
            text_column=text_column,
            provider=provider,
            as_set=True,
            output_dir=output_dir,
            output_filename=output_filename,
        )
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'images', 'videos', 'texts', or 'tabular'.")


def generate_features(
        root_folder: Union[str, Path],
        discovered_features: Optional[Dict[str, Any]] = None,
        discovered_features_path: Union[str, Path] = "outputs/discovered_features.json",
        output_dir: Union[str, Path] = "outputs",
        classes: Optional[List[str]] = None,
        provider: Optional[Any] = None,
        merge_to_single_csv: bool = True,
        merged_csv_name: str = "all_feature_values.csv",
        use_audio: bool = True,
        source_type: str = "images",
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate feature values for all images/videos in a folder using discovered features.

    This function processes images/videos organized in class subfolders and generates
    interpretable feature values for each file based on previously discovered features.

    Args:
        root_folder (Union[str, Path]): Root folder containing class subfolders with images/videos.
        discovered_features (Optional[Dict[str, Any]]): Discovered features dict from discover_features().
            If None, loads from discovered_features_path.
        discovered_features_path (Union[str, Path]): Path to discovered features JSON file.
            Default: "outputs/discovered_features.json".
        output_dir (Union[str, Path]): Directory to save output CSVs. Default: "outputs".
        classes (Optional[List[str]]): List of class names (subfolders) to process.
            If None, processes all subfolders.
        provider (Optional[Any]): LFG OpenAIProvider instance. If None, creates one from env vars.
        merge_to_single_csv (bool): Whether to merge all class CSVs into one. Default: True.
        merged_csv_name (str): Name of merged CSV file. Default: "all_feature_values.csv".
        use_audio (bool): Whether to use audio transcription (videos only). Default: True.
        source_type (str): Type of source data - "images", "videos", "texts", or "tabular". Default: "images".
        text_column (Optional[str]): Text column for tabular data.
        label_column (Optional[str]): Label column for tabular data.

    Returns:
        Dict[str, str]: Dictionary mapping class names to their output CSV paths.
            If merge_to_single_csv=True, includes "__merged__" key with merged CSV path.

    Example:
        >>> from smer_visual.smer import discover_features, generate_features
        >>> # First discover features
        >>> features = discover_features("images/sample/", source_type="images")
        >>> # Then generate feature values for all images
        >>> csv_paths = generate_features(
        ...     root_folder="images/",
        ...     discovered_features=features,
        ...     merge_to_single_csv=True
        ... )
        >>> print(csv_paths)
        {'cat': 'outputs/cat_feature_values.csv', 'dog': 'outputs/dog_feature_values.csv',
         '__merged__': 'outputs/all_feature_values.csv'}
    """
    _check_lfg_available()

    # Clean up existing output files to prevent appending to mismatched schemas (ParserError fix)
    try:
        target_output_dir = Path(output_dir)
        if target_output_dir.exists():
            # clean merged file
            if merge_to_single_csv:
                merged_path = target_output_dir / merged_csv_name
                if merged_path.exists():
                    merged_path.unlink()

            # determine classes to clean up
            classes_to_clean = classes
            if classes_to_clean is None:
                 p_root = Path(root_folder)
                 if p_root.exists() and p_root.is_dir():
                     classes_to_clean = [p.name for p in p_root.iterdir() if p.is_dir()]

            if classes_to_clean:
                for cls in classes_to_clean:
                    cls_csv = target_output_dir / f"{cls}_feature_values.csv"
                    if cls_csv.exists():
                        cls_csv.unlink()
    except Exception as e:
        print(f"Warning: Could not clean up output files: {e}")

    # If discovered_features dictionary is provided, save it to the specified path
    # because the underlying library functions expect a file path.
    if discovered_features is not None:
        import json
        out_path = Path(discovered_features_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(discovered_features, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save discovered features to {out_path}: {e}")

    if source_type == "images":
        return lfg.generate_features_from_images(
            root_folder=root_folder,
            discovered_features_path=discovered_features_path,
            output_dir=output_dir,
            classes=classes,
            provider=provider,
            merge_to_single_csv=merge_to_single_csv,
            merged_csv_name=merged_csv_name,
        )
    elif source_type == "videos":
        return lfg.generate_features_from_videos(
            root_folder=root_folder,
            discovered_features_path=discovered_features_path,
            output_dir=output_dir,
            classes=classes,
            provider=provider,
            merge_to_single_csv=merge_to_single_csv,
            merged_csv_name=merged_csv_name,
            use_audio=use_audio,
        )
    elif source_type == "texts":
        return lfg.generate_features_from_texts(
            root_folder=root_folder,
            discovered_features_path=discovered_features_path,
            output_dir=output_dir,
            classes=classes,
            provider=provider,
            merge_to_single_csv=merge_to_single_csv,
            merged_csv_name=merged_csv_name,
        )
    elif source_type == "tabular":
        if text_column is None or label_column is None:
            raise ValueError("text_column and label_column are required when source_type='tabular'")
        return lfg.generate_features_from_tabular(
            root_folder=root_folder,
            discovered_features_path=discovered_features_path,
            output_dir=output_dir,
            classes=classes,
            provider=provider,
            merge_to_single_csv=merge_to_single_csv,
            merged_csv_name=merged_csv_name,
            text_column=text_column,
            label_column=label_column,
        )
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'images', 'videos', 'texts', or 'tabular'.")


def load_lfg_features(
        csv_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load LFG-generated feature values from CSV and prepare for classification.

    Args:
        csv_path (Union[str, Path]): Path to the CSV file generated by generate_features().
        feature_columns (Optional[List[str]]): List of feature column names to use.
            If None, auto-detects feature columns (excludes 'Image', 'File', 'Class', 'raw_llm_output').

    Returns:
        pd.DataFrame: DataFrame with Image, Class, and feature columns ready for classification.

    Example:
        >>> from smer_visual.smer import load_lfg_features
        >>> df = load_lfg_features("outputs/all_feature_values.csv")
        >>> print(df.columns)
        Index(['Image', 'Class', 'color', 'texture', 'shape', ...])
    """
    df = pd.read_csv(csv_path)

    if feature_columns is None:
        # Auto-detect feature columns (exclude metadata columns)
        exclude_cols = {'Image', 'File', 'Class', 'raw_llm_output'}
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Extract just the value part from "feature_name = value" format
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).split(' = ', 1)[-1] if pd.notna(x) else x)

    return df


def embed_lfg_features(
        df: pd.DataFrame,
        embedding_model: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate embeddings for LFG feature values, similar to embed_descriptions for SMER.

    Each feature value is embedded separately, allowing SMER-style perturbation analysis
    on LFG features. The result is a DataFrame compatible with SMER classification functions.

    Args:
        df (pd.DataFrame): DataFrame from load_lfg_features() with feature columns.
        embedding_model (Union[str, Path]): Model identifier for OpenAI or path to local model.
        feature_columns (Optional[List[str]]): Feature columns to embed.
            If None, auto-detects feature columns.
        api_key (Optional[str]): OpenAI API key (required for OpenAI embeddings).

    Returns:
        pd.DataFrame: DataFrame with columns: image, description, embedding, label.
            - 'description': concatenated feature values as "feature1:value1 feature2:value2 ..."
            - 'embedding': list of embeddings, one per feature value

    Example:
        >>> from smer_visual.smer import load_lfg_features, embed_lfg_features
        >>> df = load_lfg_features("outputs/all_feature_values.csv")
        >>> df_embedded = embed_lfg_features(df, embedding_model="text-embedding-3-small", api_key="...")
        >>> # Now use with classify_lr(), plot_aopc(), etc.
    """
    OPENAI_MODELS = {'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'}

    if feature_columns is None:
        exclude_cols = {'Image', 'File', 'Class', 'raw_llm_output'}
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    results = []

    def get_openai_embedding(client, text, model):
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def get_local_embedding(tokenizer, model, device, text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return np.squeeze(outputs.last_hidden_state[:, 0, :].cpu().numpy())

    # Initialize model based on type
    client = None
    tokenizer = None
    local_model = None
    device = None
    use_openai = False

    if isinstance(embedding_model, str) and embedding_model in OPENAI_MODELS:
        if not api_key:
            raise ValueError("API key required for OpenAI embeddings")
        client = OpenAI(api_key=api_key)
        use_openai = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        local_model = AutoModel.from_pretrained(embedding_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_model.to(device)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding LFG features"):
        # Build description string from feature values
        feature_tokens = []
        embeddings = []

        for col in feature_columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                # Create token as "feature_name:value"
                # Keep original with spaces for embedding
                token_original = f"{col}:{value}"

                # Create sanitized version for description (replace separate words with underscores)
                # This ensures description.split() keeps the feature atomic
                col_sanitized = str(col).replace(" ", "_")
                val_sanitized = str(value).strip().replace(" ", "_")
                token_sanitized = f"{col_sanitized}:{val_sanitized}"

                feature_tokens.append(token_sanitized)

                try:
                    if use_openai:
                        emb = get_openai_embedding(client, token_original, embedding_model)
                    else:
                        emb = get_local_embedding(tokenizer, local_model, device, token_original)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Error embedding '{token_original}': {e}")
                    embeddings.append(None)

        # Filter out None embeddings
        valid_pairs = [(t, e) for t, e in zip(feature_tokens, embeddings) if e is not None]
        if valid_pairs:
            feature_tokens, embeddings = zip(*valid_pairs)
            feature_tokens = list(feature_tokens)
            embeddings = list(embeddings)
        else:
            feature_tokens = []
            embeddings = []

        description = ' '.join(feature_tokens)

        results.append({
            'image': row.get('Image', idx),
            'description': description,
            'embedding': embeddings if embeddings else None,
            'label': row.get('Class', None),
            'feature_names': feature_columns,  # Store feature names for reference
        })

    return pd.DataFrame(results)


def _predict_proba_for_lfg_features(feature_mask: List[bool], row: pd.Series,
                                     logreg_model: LogisticRegression) -> np.ndarray:
    """
    Predict probabilities for LFG features with a mask indicating which features to include.

    Args:
        feature_mask (List[bool]): Boolean mask for which features to include.
        row (pd.Series): Row containing embeddings.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    embeddings = row['embedding']
    if embeddings is None or len(embeddings) == 0:
        return np.zeros(len(logreg_model.classes_))

    # Select embeddings based on mask
    selected_embeddings = [emb for emb, mask in zip(embeddings, feature_mask) if mask]

    if len(selected_embeddings) > 0:
        emb = np.mean(selected_embeddings, axis=0)
    else:
        emb = np.zeros(len(embeddings[0]))

    # Compute logits
    logits = np.dot(emb, logreg_model.coef_.T) + logreg_model.intercept_
    logits = np.asarray(logits).ravel()

    # Softmax/sigmoid based on number of classes
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_max = np.max(x)
        ex = np.exp(x - x_max)
        return ex / ex.sum()

    if logreg_model.coef_.shape[0] > 1:
        probs = _softmax(logits)
    else:
        logit_val = float(logits[0]) if logits.size > 0 else 0.0
        p = 1.0 / (1.0 + np.exp(-logit_val))
        probs = np.array([1.0 - p, p])

    return probs


def classify_lr_lfg(
        dataset: pd.DataFrame,
        X_train: np.ndarray,
        logreg_model: LogisticRegression,
        test_idx=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply SMER classification method to LFG-embedded features.

    This is the LFG equivalent of classify_lr(). It computes feature importance
    by measuring probability drop when each feature is removed (SMER perturbation method).

    Args:
        dataset (pd.DataFrame): DataFrame from embed_lfg_features() with embeddings.
        X_train (np.ndarray): Training data embeddings.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (AOPC DataFrame, updated dataset with importance scores)

    Example:
        >>> from smer_visual.smer import load_lfg_features, embed_lfg_features, classify_lr_lfg
        >>> df = load_lfg_features("outputs/all_feature_values.csv")
        >>> df_embedded = embed_lfg_features(df, "text-embedding-3-small", api_key="...")
        >>> X_train = np.vstack([aggregate_embeddings(e) for e in df_embedded['embedding'][:80]])
        >>> logreg_model = LogisticRegression().fit(X_train, df_embedded['label'][:80])
        >>> df_aopc, df_updated = classify_lr_lfg(df_embedded, X_train, logreg_model)
    """
    dataset = dataset.copy()
    dataset['feature_importance'] = None
    dataset['sorted_features_by_importance'] = None

    for idx in range(len(dataset)):
        row = dataset.iloc[idx]
        embeddings = row['embedding']

        if embeddings is None or len(embeddings) == 0:
            continue

        description = row['description']
        tokens = description.split() if description else []
        n_features = len(embeddings)

        if n_features == 0:
            continue

        # Get original prediction with all features
        all_mask = [True] * n_features
        original_probs = _predict_proba_for_lfg_features(all_mask, row, logreg_model)

        target_class = row['label']
        if target_class in logreg_model.classes_.tolist():
            class_index = logreg_model.classes_.tolist().index(target_class)
        else:
            class_index = 0

        importance_scores = {}

        # Compute importance by removing each feature
        for i in range(n_features):
            # Create mask with feature i removed
            perturbed_mask = [True] * n_features
            perturbed_mask[i] = False

            perturbed_probs = _predict_proba_for_lfg_features(perturbed_mask, row, logreg_model)
            drop = original_probs[class_index] - perturbed_probs[class_index]

            feature_token = tokens[i] if i < len(tokens) else f"feature_{i}"
            importance_scores[feature_token] = drop

        # Sort by importance
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        importance_str = ','.join(f"{feat}:{score:.4f}" for feat, score in sorted_importance)
        dataset.at[idx, 'feature_importance'] = importance_str

        sorted_features_list = [feat for feat, score in sorted_importance]
        dataset.at[idx, 'sorted_features_by_importance'] = ','.join(sorted_features_list)

    # Create AOPC dataset (test set portion)
    if test_idx is not None:
        df_aopc = dataset.iloc[test_idx].reset_index(drop=True)
    else:
        df_aopc = dataset[X_train.shape[0]:].reset_index(drop=True)
    return df_aopc, dataset


def compute_aopc_lfg(df: pd.DataFrame, top_features: List[str], max_k: int,
                      logreg_model: LogisticRegression) -> List[float]:
    """
    Compute AOPC (Area Over Perturbation Curve) for LFG features.

    This is the LFG equivalent of compute_aopc(), measuring the average probability
    drop when top features are progressively removed.

    Args:
        df (pd.DataFrame): DataFrame with LFG embeddings.
        top_features (List[str]): List of top feature tokens to remove.
        max_k (int): Maximum number of features to remove.
        logreg_model (LogisticRegression): Pre-trained logistic regression model.

    Returns:
        List[float]: Average probability drops for K=0,1,...,max_k.
    """
    avg_drops = []

    for K in range(0, max_k + 1):
        drops = []

        for idx, row in df.iterrows():
            embeddings = row['embedding']
            description = row['description']

            if embeddings is None or len(embeddings) == 0:
                continue

            tokens = description.split() if description else []
            n_features = len(embeddings)

            # Original prediction
            all_mask = [True] * n_features
            original_probs = _predict_proba_for_lfg_features(all_mask, row, logreg_model)
            original_class = np.argmax(original_probs)
            original_prob = original_probs[original_class]

            # Find which top_features are in this sample's tokens
            top_in_sample = [t for t in top_features if t in tokens]
            features_to_remove = top_in_sample[:K]

            if not features_to_remove:
                drop = 0.0
            else:
                # Create mask removing these features
                perturbed_mask = [
                    tokens[i] not in features_to_remove
                    for i in range(n_features)
                ]
                perturbed_probs = _predict_proba_for_lfg_features(perturbed_mask, row, logreg_model)
                perturbed_prob = perturbed_probs[original_class]
                drop = original_prob - perturbed_prob

            drops.append(drop)

        avg_drops.append(np.mean(drops) if drops else 0.0)

    return avg_drops


def build_custom_predict_lfg(row: pd.Series, logreg_model: LogisticRegression):
    """
    Build a LIME-compatible prediction function for LFG features.

    Args:
        row (pd.Series): Row with LFG embeddings.
        logreg_model (LogisticRegression): Pre-trained model.

    Returns:
        Callable: Prediction function for LIME.
    """
    def predict_for_lime(texts):
        emb_list = []
        original_tokens = row['description'].split() if row['description'] else []
        embeddings = row['embedding'] if row['embedding'] is not None else []
        token_to_index = {t: i for i, t in enumerate(original_tokens)}

        for t in texts:
            t_tokens = t.split()
            valid_indices = [token_to_index[x] for x in t_tokens if x in token_to_index]

            if len(valid_indices) > 0 and len(embeddings) > 0:
                emb = np.mean([np.array(embeddings[ix]) for ix in valid_indices], axis=0)
            else:
                if len(embeddings) > 0:
                    emb = np.zeros(len(embeddings[0]))
                else:
                    emb = np.zeros(logreg_model.coef_.shape[1])
            emb_list.append(emb)

        return logreg_model.predict_proba(emb_list)

    return predict_for_lime


def plot_aopc_lfg(df_aopc: pd.DataFrame, logreg_model: LogisticRegression, max_k: int = 6):
    """
    Plot AOPC curves comparing SMER and LIME importance for LFG features.

    This is the LFG equivalent of plot_aopc(), providing interpretability analysis
    on LFG-generated features using the SMER perturbation method.

    Args:
        df_aopc (pd.DataFrame): DataFrame with LFG embeddings (test set).
        logreg_model (LogisticRegression): Pre-trained logistic regression model.
        max_k (int): Maximum number of features to remove. Default: 6.

    Returns:
        None: Displays the plot.

    Example:
        >>> from smer_visual.smer import plot_aopc_lfg
        >>> plot_aopc_lfg(df_aopc, logreg_model, max_k=6)
    """
    # SMER importance calculation for LFG features
    smer_rows = []

    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing SMER importances (LFG)"):
        embeddings = row['embedding']
        description = row['description']

        if embeddings is None or len(embeddings) == 0:
            continue

        tokens = description.split() if description else []
        n_features = len(embeddings)

        # Original prediction
        all_mask = [True] * n_features
        original_probs = _predict_proba_for_lfg_features(all_mask, row, logreg_model)
        original_class = np.argmax(original_probs)
        original_prob = original_probs[original_class]

        # Compute importance for each feature
        for i in range(n_features):
            perturbed_mask = [True] * n_features
            perturbed_mask[i] = False

            perturbed_probs = _predict_proba_for_lfg_features(perturbed_mask, row, logreg_model)
            drop = original_prob - perturbed_probs[original_class]

            feature_token = tokens[i] if i < len(tokens) else f"feature_{i}"
            smer_rows.append({
                'feature': feature_token,
                'importance': abs(drop)
            })

    smer_df = pd.DataFrame(smer_rows)

    # LIME importance calculation
    if 'label' in df_aopc.columns:
        class_names = df_aopc['label'].unique().tolist()
    else:
        class_names = []

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
    lime_rows = []

    for idx, row in tqdm(df_aopc.iterrows(), total=len(df_aopc), desc="Computing LIME importances (LFG)"):
        description = row['description']
        if not description or row['embedding'] is None:
            continue

        lime_predict_fn = build_custom_predict_lfg(row, logreg_model)

        try:
            exp = explainer.explain_instance(
                text_instance=description,
                classifier_fn=lime_predict_fn,
                num_features=len(description.split())
            )

            for feat, imp in exp.as_list():
                lime_rows.append({
                    'feature': feat,
                    'importance': abs(imp)
                })
        except Exception as e:
            print(f"LIME error for row {idx}: {e}")
            continue

    lime_df = pd.DataFrame(lime_rows)

    # Aggregate and get top features
    if len(smer_df) > 0:
        global_importances_smer = (
            smer_df.groupby('feature')['importance'].mean()
            .reset_index()
            .sort_values('importance', ascending=False)
        )
        top_features_smer = global_importances_smer['feature'].head(20).tolist()
    else:
        top_features_smer = []

    if len(lime_df) > 0:
        global_importances_lime = (
            lime_df.groupby('feature')['importance'].mean()
            .reset_index()
            .sort_values('importance', ascending=False)
        )
        top_features_lime = global_importances_lime['feature'].head(20).tolist()
    else:
        top_features_lime = []

    # Compute AOPC
    AOPC_SMER = compute_aopc_lfg(df_aopc, top_features_smer, max_k, logreg_model)
    AOPC_LIME = compute_aopc_lfg(df_aopc, top_features_lime, max_k, logreg_model)

    # Plot
    plt.figure(figsize=(10, 6))
    x_values = range(0, max_k + 1)
    plt.plot(x_values, AOPC_SMER, marker='o', label='SMER (LFG Features)')
    plt.plot(x_values, AOPC_LIME, marker='x', label='LIME (LFG Features)')

    plt.xlabel('Number of Features Removed (K)')
    plt.ylabel('Average Probability Drop')
    plt.title('AOPC Comparison: SMER vs LIME on LFG Features')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_important_features_lfg(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Plot the top 10 most important LFG features across all samples.

    This is the LFG equivalent of plot_important_words().

    Args:
        dataset (pd.DataFrame): DataFrame with 'sorted_features_by_importance' column.

    Returns:
        pd.DataFrame: DataFrame with top 10 features and their counts.

    Example:
        >>> from smer_visual.smer import plot_important_features_lfg
        >>> top_features = plot_important_features_lfg(dataset)
    """
    most_important_features = []

    for idx in range(len(dataset)):
        sorted_features = dataset.at[idx, 'sorted_features_by_importance']
        if sorted_features and isinstance(sorted_features, str):
            first_feature = sorted_features.split(',')[0].strip()
            if first_feature:
                most_important_features.append(first_feature)

    feature_counts = pd.Series(most_important_features).value_counts().reset_index()
    feature_counts.columns = ['feature', 'count']

    top_10_features = feature_counts.head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_10_features, x='feature', y='count', palette='viridis', hue='count')
    plt.title('Top 10 Most Important LFG Features Across Samples')
    plt.xlabel('Feature (name:value)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend([], [], frameon=False)
    plt.show()

    return top_10_features


def smer_lfg_pipeline(
        data_folder: Union[str, Path],
        embedding_model: Union[str, Path],
        model: str = "gpt-4o-mini",
        discovery_samples: Union[str, Path, List[str], None] = None,
        api_key: Optional[str] = None,
        source_type: str = "images",
        output_dir: Union[str, Path] = "outputs",
        provider: Optional[Any] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_audio: bool = True,
        max_k: int = 6,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end SMER pipeline using LFG features for interpretable classification.

    This pipeline combines the LFG feature discovery/generation with the SMER
    perturbation-based interpretability method. It:
    1. Discovers features from sample images/videos using LFG
    2. Generates feature values for all data using LFG
    3. Embeds feature values for SMER analysis
    4. Trains classifier and computes SMER importance scores
    5. Generates AOPC plots comparing SMER vs LIME

    Args:
        data_folder (Union[str, Path]): Root folder with class subfolders.
        embedding_model (Union[str, Path]): Embedding model for feature values.
        model (str): Model for description generation (e.g. "gpt-4o-mini").
        discovery_samples (Union[str, Path, List[str], None]): Samples for feature discovery.
        api_key (Optional[str]): API key for OpenAI embeddings.
        source_type (str): "images" or "videos". Default: "images".
        output_dir (Union[str, Path]): Output directory. Default: "outputs".
        provider (Optional[Any]): LFG OpenAIProvider instance.
        test_size (float): Test set fraction. Default: 0.2.
        random_state (int): Random seed. Default: 42.
        use_audio (bool): Use audio for videos. Default: True.
        max_k (int): Max features to remove for AOPC. Default: 6.
        text_column (Optional[str]): Text column for tabular data.
        label_column (Optional[str]): Label column for tabular data.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'discovered_features': Discovered feature specification
            - 'feature_values_path': Path to generated CSV
            - 'embedded_data': DataFrame with embeddings
            - 'model': Trained classifier
            - 'df_aopc': AOPC analysis DataFrame
            - 'dataset': Full dataset with importance scores
            - 'top_features': Top important features DataFrame

    Example:
        >>> from smer_visual.smer import smer_lfg_pipeline
        >>> results = smer_lfg_pipeline(
        ...     data_folder="images/",
        ...     embedding_model="text-embedding-3-small",
        ...     api_key="your_api_key",
        ...     source_type="images"
        ... )
        >>> print(results['top_features'])
    """
    _check_lfg_available()

    # Initialize provider if not given but api_key is provided
    if provider is None and api_key:
        if hasattr(lfg, 'OpenAIProvider'):
            provider = lfg.OpenAIProvider(api_key=api_key, default_deployment_name=model)
        else:
            # Fallback to direct import
            from llm_feature_gen.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=api_key, default_deployment_name=model)

    from sklearn.model_selection import train_test_split

    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover features using LFG
    print(f"Step 1: Discovering features with LFG using model '{model}'...")
    if discovery_samples is None:
        if not data_folder.exists():
            raise ValueError(f"Data folder '{data_folder.absolute()}' does not exist.")

        if source_type == "tabular":
            # For tabular, we can pass the directory or file directly to discovery.
            # llm-feature-gen will sample internally.
            discovery_samples = data_folder
        else:
            ext_map = {
                "images": {".jpg", ".jpeg", ".png"},
                "videos": {".mp4", ".mov", ".avi", ".mkv"},
                "texts": {".txt", ".md", ".json"}
            }
            valid_exts = ext_map.get(source_type, set())
            sample_files = []

            # Check if folder structure is valid (class subfolders)
            subfolders = [d for d in data_folder.iterdir() if d.is_dir()]
            if not subfolders:
                 print(f"Warning: No subfolders found in '{data_folder}'. Expected structure: data_folder/class_name/file.ext")

            for subfolder in subfolders:
                folder_files = [f for f in subfolder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
                if folder_files:
                    sample_files.extend([str(f) for f in folder_files[:5]])
                    if len(sample_files) >= 5:
                        break

            # Safety check
            if not sample_files:
                # Fallback: check root folder
                root_files = [str(f) for f in data_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
                if root_files:
                     print(f"KeyInfo: Found {len(root_files)} files in root folder, using up to 5 for discovery.")
                     sample_files = root_files[:5]
                else:
                     raise ValueError(f"No valid files found in '{data_folder}' or its subfolders with extensions {valid_exts}")

            print(f"Using {len(sample_files)} sample files for feature discovery: {sample_files}")
            discovery_samples = sample_files

    discovered_features = discover_features(
        data_source=discovery_samples,
        source_type=source_type,
        provider=provider,
        output_dir=output_dir,
        use_audio=use_audio,
        text_column=text_column,
    )

    num_features = len(discovered_features.get('proposed_features', []))
    print(f"Discovered {num_features} features")

    if num_features == 0:
        print("Warning: 0 features were discovered. This usually indicates an issue with the LLM provider or the model.")
        print(f"{discovered_features = }")

        # Attempt to see if there was an error message in the return dict (if lfg adds it)
        if 'error' in discovered_features:
             print(f"Provider Error: {discovered_features['error']}")
        elif 'features' in discovered_features and isinstance(discovered_features['features'], str):
             # This happens when the provider returns text that couldn't be parsed as JSON
             print(f"Provider raw (fallback) response: {discovered_features['features']}")
        elif not discovered_features:
             print("Provider returned completely empty result.")

    # Step 2: Generate feature values
    print("Step 2: Generating feature values with LFG...")
    csv_paths = generate_features(
        root_folder=data_folder,
        discovered_features=discovered_features,
        output_dir=output_dir,
        provider=provider,
        merge_to_single_csv=True,
        source_type=source_type,
        use_audio=use_audio,
        text_column=text_column,
        label_column=label_column,
    )
    merged_csv_path = csv_paths.get('__merged__', list(csv_paths.values())[0])
    print(f"Feature values saved to: {merged_csv_path}")

    # Step 3: Load and embed features for SMER
    print("Step 3: Loading and embedding LFG features...")
    df = load_lfg_features(merged_csv_path)
    df_embedded = embed_lfg_features(df, embedding_model, api_key=api_key)
    print(f"Embedded {len(df_embedded)} samples")

    # Step 4: Train classifier
    print("Step 4: Training classifier...")
    # Filter out samples with no embeddings
    valid_mask = df_embedded['embedding'].apply(lambda x: x is not None and len(x) > 0)
    df_valid = df_embedded[valid_mask].reset_index(drop=True)

    X = np.vstack([aggregate_embeddings(e) for e in df_valid['embedding']])
    y = df_valid['label'].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, range(len(X)), test_size=test_size, random_state=random_state, stratify=y
    )

    logreg_model = LogisticRegression(max_iter=1000, random_state=random_state)
    logreg_model.fit(X_train, y_train)

    accuracy = logreg_model.score(X_test, y_test)
    print(f"Classifier accuracy: {accuracy:.4f}")

    # Step 5: SMER analysis
    print("Step 5: Computing SMER feature importance...")
    df_aopc, dataset = classify_lr_lfg(df_valid, X_train, logreg_model, test_idx=test_idx)

    # Step 6: Plot results
    print("Step 6: Generating plots...")
    plot_aopc_lfg(df_aopc, logreg_model, max_k=max_k)
    top_features = plot_important_features_lfg(dataset)

    return {
        'discovered_features': discovered_features,
        'feature_values_path': merged_csv_path,
        'embedded_data': df_embedded,
        'model': logreg_model,
        'df_aopc': df_aopc,
        'dataset': dataset,
        'top_features': top_features,
        'accuracy': accuracy,
    }


def smer_text_pipeline(
    data_folder: Union[str, Path],
    model: str = "gpt-4o-mini",
    embedding_model: Union[str, Path] = "text-embedding-ada-002",
    api_key: Optional[str] = None,
    user_prompt: str = "Describe this image in 7 words. Be concise, try to maximize the information about the objects in this image.",
    output_dir: Union[str, Path] = "outputs",
    test_size: float = 0.2,
    random_state: int = 42,
    max_k: int = 6,
) -> Dict[str, Any]:
    """
    End-to-end SMER pipeline using text descriptions.

    This pipeline runs the standard SMER workflow:
    1. Generate text descriptions for images
    2. Embed descriptions
    3. Train classifier
    4. Compute SMER word importance (AOPC)
    5. Generate plots

    Args:
        data_folder (Union[str, Path]): Root folder with class subfolders.
        model (str): Model for description generation (e.g. "gpt-4o-mini").
        embedding_model (Union[str, Path]): Model for embeddings.
        api_key (Optional[str]): OpenAI API key.
        user_prompt (str): Prompt for description generation.
        output_dir (Union[str, Path]): Output directory.
        test_size (float): Test set fraction.
        random_state (int): Random seed.
        max_k (int): Max words to remove for AOPC.

    Returns:
        Dict[str, Any]: Dictionary containing results.
    """
    from sklearn.model_selection import train_test_split

    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate Descriptions
    print("Step 1: Generating image descriptions...")
    descriptions = image_descriptions(
        model=model,
        data_folder=data_folder,
        api_key=api_key,
        user_prompt=user_prompt
    )

    # Step 2: Embed Descriptions
    print("Step 2: Embedding descriptions...")
    df_embedded = embed_descriptions(
        descriptions=descriptions,
        embedding_model=embedding_model,
        api_key=api_key
    )
    print(f"Embedded {len(df_embedded)} samples")

    # Step 3: Train Classifier
    print("Step 3: Training classifier...")
    # Filter valid
    valid_mask = df_embedded['embedding'].apply(lambda x: x is not None and len(x) > 0)
    df_valid = df_embedded[valid_mask].reset_index(drop=True)

    X = np.vstack([aggregate_embeddings(e) for e in df_valid['embedding']])
    y = df_valid['label'].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, range(len(X)), test_size=test_size, random_state=random_state, stratify=y
    )

    logreg_model = LogisticRegression(max_iter=1000, random_state=random_state)
    logreg_model.fit(X_train, y_train)

    accuracy = logreg_model.score(X_test, y_test)
    print(f"Classifier accuracy: {accuracy:.4f}")

    # Step 4: SMER Analysis
    print("Step 4: Computing SMER word importance...")
    df_aopc, dataset = classify_lr(df_valid, X_train, logreg_model, test_idx=test_idx)

    # Step 5: Plot Results
    print("Step 5: Generating plots...")
    plot_aopc(df_aopc, logreg_model, max_k=max_k)
    top_words = plot_important_words(dataset)

    return {
        'descriptions': descriptions,
        'embedded_data': df_embedded,
        'model': logreg_model,
        'df_aopc': df_aopc,
        'dataset': dataset,
        'top_words': top_words,
        'accuracy': accuracy,
    }

def smer_pipeline(
    data_folder: Union[str, Path],
    mode: str = "text",
    # Text mode args
    model: str = "gpt-4o-mini",
    user_prompt: str = "Describe this image in 7 words. Be concise.",
    # LLM features mode args
    discovery_samples: Union[str, Path, List[str], None] = None,
    provider: Optional[Any] = None,
    source_type: str = "images",
    # Common args
    embedding_model: Union[str, Path] = "text-embedding-ada-002",
    api_key: Optional[str] = None,
    output_dir: Union[str, Path] = "outputs",
    test_size: float = 0.2,
    random_state: int = 42,
    use_audio: bool = True,
    max_k: int = 6,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified SMER pipeline supporting both text descriptions and LLM-generated features.

    Args:
        mode (str): "text" for standard description-based SMER,
                    "llm_features" for LFG-based SMER.
        ... (other args passed to respective pipelines) ...
    """
    if mode == "text":
        return smer_text_pipeline(
            data_folder=data_folder,
            model=model,
            embedding_model=embedding_model,
            api_key=api_key,
            user_prompt=user_prompt,
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            max_k=max_k,
        )
    elif mode == "llm_features":
        return smer_lfg_pipeline(
            data_folder=data_folder,
            embedding_model=embedding_model,
            model=model,
            discovery_samples=discovery_samples,
            api_key=api_key,
            source_type=source_type,
            output_dir=output_dir,
            provider=provider,
            test_size=test_size,
            random_state=random_state,
            use_audio=use_audio,
            max_k=max_k,
            text_column=text_column,
            label_column=label_column,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'text' or 'llm_features'.")
