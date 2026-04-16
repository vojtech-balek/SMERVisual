# API Reference

## `smer_visual.smer`

### `image_descriptions(model, data_folder, api_key=None, user_prompt=...)`

Generates image descriptions for each file found in `data_folder`.

- OpenAI models require `api_key`
- Local model identifiers or paths are also supported
- Returns a dictionary keyed by image path with `label`, `description`, and `error`

### `embed_descriptions(descriptions, embedding_model, api_key=None)`

Generates per-word embeddings for descriptions returned by `image_descriptions()`.

- OpenAI embedding models require `api_key`
- Local embedding models are supported through `transformers`
- Returns a `pandas.DataFrame` with `image`, `description`, `embedding`, and `label`

### `aggregate_embeddings(embedding_list)`

Computes the mean vector over a list of embeddings.

### `classify_lr(dataset, X_train, logreg_model)`

Computes SMER-style feature importance for each description using a trained logistic regression model.

Returns:

- `df_aopc`: rows reserved for AOPC-style evaluation
- `dataset`: the input dataset enriched with:
  `feature_importance`, `sorted_words_by_importance`, and `sorted_words_by_importance_processed`

### `compute_aopc(df, top_words, max_k, logreg_model)`

Computes average probability drops when removing the top `K` words from each description.

### `build_custom_predict(row, logreg_model)`

Creates a prediction function compatible with `lime.lime_text.LimeTextExplainer`.

### `plot_aopc(df_aopc, logreg_model, max_k=6)`

Plots AOPC curves for SMER and LIME.

### `plot_important_words(dataset)`

Plots the most frequent top-ranked important words across the dataset and returns a dataframe of counts.

### `_bounding_boxes(image_path, df_top_words, model_id, ...)`

Internal helper used to draw predicted zero-shot object-detection boxes for selected words.

### `save_bounding_box_images(input_path, output_folder, df_top_words, model_id=..., ...)`

Runs bounding box generation on a single image or a directory and writes annotated copies to `output_folder`.

Returns a dictionary mapping each input image path to its saved output path.

## `smer_visual.utils`

### `_get_image_files_with_class(folder_path)`

Yields `(image_path, class_name)` pairs based on parent directory names.

### `_encode_image(image_path)`

Loads an image file and returns a base64-encoded string.

### `_preprocess_text(text)`

Normalizes comma-separated text with spaCy lemmatization.
