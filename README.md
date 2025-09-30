# **SMERVisual**

SMERVisual is a Python package for **explainable machine learning** using the **Self Model Entities Related (SMER) method**.  
It provides tools for classifying images with **LLM-generated text descriptions**, followed by explainability analysis using SMER.

The package supports both **OpenAI API models** and **local models**, giving you flexibility in how you run it.

---

## Installation

Install SMERVisual via pip:

```sh
pip install smer-visual
```
---
## **Data Preparation**
Before using SMERVisual, you must prepare your dataset in the following structure:

```kotlin
data/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
```


- Each subfolder name is treated as a class label.
- Images must be placed inside their respective class folders.

Recommended minimum: At least 20 images per class for meaningful results.

## **API Keys and Configuration**
For OpenAI-based models, you need an API key.
Best practice is to store your API key in a .env file and load it as an environment variable:
1. Create a `.env` file in your project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
2. Load the environment variable in your script:
   ```python
   from dotenv import load_dotenv
   import os 
   load_dotenv()
   api_key = os.getenv("OPENAI_API_KEY")
   ```
3. Pass api_key to functions that require it.

4. Do not hardcode API keys directly in scripts.

## Quick Start (Working Example)
Below is a complete example that demonstrates the correct workflow: 
- Generating image descriptions
- Embedding descriptions
- Splitting data into train/test sets 
- Training logistic regression
- Running SMER explainability

```python
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from smer_visual.smer import (
    image_descriptions,
    embed_descriptions,
    aggregate_embeddings,
    classify_lr,
    plot_important_words,
    plot_aopc,
)

# 1. Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. Generate image descriptions
descriptions = image_descriptions(
    model="gpt-4o-mini",
    data_folder="data/",
    api_key=api_key
)

# 3. Generate embeddings
embeddings_df = embed_descriptions(
    descriptions=descriptions,
    embedding_model="text-embedding-ada-002",
    api_key=api_key
)

# 4. Aggregate embeddings
embeddings_df["aggregated_embedding"] = embeddings_df["embedding"].apply(aggregate_embeddings)

# 5. Split into train/test
X = np.stack(embeddings_df["aggregated_embedding"].values)
y = embeddings_df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Train a logistic regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# 7. Run SMER classification and explanations
aopc_df, updated_dataset = classify_lr(embeddings_df, X_test, logreg_model)

# 8. Visualizations
top_words = plot_important_words(updated_dataset)
plot_aopc(aopc_df, logreg_model, max_k=5)

# 9. Save bounding box images 
from smer_visual.smer import save_bounding_box_images
save_bounding_box_images(
    input_path="data/",
    output_folder="output",
    df_top_words=top_words_df,
    model_id="IDEA-Research/grounding-dino-base",
    box_threshold = 0.5,
    text_threshold = 0.4
)
```
---
---

## **Why Use SMERVisual?**

- **Explainable AI** – Understand model decisions with SMER.
- **Model-Agnostic** – Compatible with OpenAI APIs and open-source models.
- **Zero-Shot Detection** – No extra training data required.
- **Easy Integration** – Simple API, modular design

---

## **Contributing**
Contributions are welcome. If you’d like to contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## **License**
SMERVisual is released under the **MIT License**.

---
