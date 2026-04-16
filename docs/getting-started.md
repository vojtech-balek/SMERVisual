# Getting Started

## Installation

Install the package from PyPI:

```bash
pip install smer-visual
```

The package requires Python 3.10 or newer.

## Input Data Layout

`image_descriptions()` expects images to be organized in subfolders where each folder name is the class label:

```text
dataset/
  cats/
    cat-1.jpg
    cat-2.png
  dogs/
    dog-1.jpg
```

## Quickstart

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from smer_visual.smer import (
    aggregate_embeddings,
    classify_lr,
    embed_descriptions,
    image_descriptions,
    plot_aopc,
    plot_important_words,
)

descriptions = image_descriptions(
    model="gpt-4o-mini",
    data_folder="dataset",
    api_key="your-openai-api-key",
)

embeddings_df = embed_descriptions(
    descriptions=descriptions,
    embedding_model="text-embedding-3-small",
    api_key="your-openai-api-key",
)

embeddings_df["aggregated_embedding"] = embeddings_df["embedding"].apply(aggregate_embeddings)
X = np.stack(embeddings_df["aggregated_embedding"].values)
y = embeddings_df["label"].to_numpy()

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

df_aopc, explained = classify_lr(embeddings_df.copy(), X, model)
top_words = plot_important_words(explained)
plot_aopc(df_aopc, model, max_k=5)
```

## Bounding Boxes

After identifying important words, you can save annotated images:

```python
from smer_visual.smer import save_bounding_box_images

results = save_bounding_box_images(
    input_path="dataset",
    output_folder="output",
    df_top_words=top_words,
)
```

## Notes

- Use an OpenAI API key only when you choose an OpenAI model.
- Local model paths are also supported for descriptions and embeddings.
- The package imports heavy ML dependencies, so installation is best done in an isolated environment.
