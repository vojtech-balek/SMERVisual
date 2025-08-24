# **SMERVisual**

SMERVisual is a Python package designed for explainable machine learning using the **Self Model Entities Related (SMER) method**. It provides tools for explainable classification of images with **LLM-generated text descriptions**, which are then analyzed using the SMER explanation technique.

The package supports both **OpenAI API** models and **local language models**, offering flexibility in model selection.

---

## **Installation**
Install SMERVisual using pip:
```sh
pip install smer-visual
```

---

## **Features**

### **Image Description**
The `image_description` function generates text descriptions for images using either OpenAI models or local language models. Key features include:
- Support for OpenAI models like `gpt-4o-mini` and local models.
- Customizable prompts for generating concise and informative descriptions.

### **Description Embeddings**
The `get_description_embeddings` function computes embeddings for image descriptions using OpenAI or local models. 

### **Explainable Classification**
The `classify_with_logreg` function performs logistic regression-based classification on image datasets while computing **SMER** values for explainability. Key features include:
- Aggregation of embeddings for classification.
- Computation of feature importance for each word in the description.
- Support for **AOPC (Area Over the Perturbation Curve)** analysis.

### **Visualization**
- **Important Words**: The `plot_important_words` function visualizes the most important words across images.
- **AOPC Curves**: The `plot_aopc` function compares the explainability of SMER and LIME methods by plotting AOPC curves.
- **Bounding Boxes**: The `BoundingBoxGenerator` class overlays bounding boxes on images, highlighting critical words identified in classification.

---

## **Usage Example**

### **Image Description and Embeddings**
```python
from smer_visual.smer import image_description, get_description_embeddings

# Generate image descriptions
descriptions = image_description(
    model="gpt-4o-mini",
    data_folder="path/to/images",
    api_key="your_openai_api_key"
)

# Generate embeddings for descriptions
embeddings_df = get_description_embeddings(
    descriptions=descriptions,
    embedding_model="text-embedding-ada-002",
    api_key="your_openai_api_key"
)
```

### **Explainable Classification**
```python
from smer_visual.smer import classify_with_logreg, aggregate_embeddings
from sklearn.linear_model import LogisticRegression
import numpy as np

# Prepare data
embeddings_df["aggregated_embedding"] = embeddings_df["embedding"].apply(aggregate_embeddings)
X_train = np.stack(embeddings_df["aggregated_embedding"].values)
y_train = embeddings_df["label"]

# Train a logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Perform classification and compute feature importance
aopc_df, updated_dataset = classify_with_logreg(embeddings_df, X_train, logreg_model)
```

### **Visualization**
```python
from smer_visual.smer import plot_important_words, plot_aopc

# Plot important words
plot_important_words(updated_dataset)

# Plot AOPC curves
plot_aopc(aopc_df, logreg_model, max_k=5)
```

### **Bounding Box Generation**
```python
from smer_visual.smer import save_bounding_box_images
results = save_bounding_box_images(
    input_path="data/",
    output_folder="output",
    df_top_words=top_words_df,
    model_id="IDEA-Research/grounding-dino-base",
    box_threshold = 0.5,
    text_threshold = 0.4
)

```

---

## **Why Use SMERVisual?**

- **Explainable AI** – Provides insight into model decision-making.
- **Model-Agnostic** – Compatible with OpenAI APIs and open-source models.
- **Zero-Shot Detection** – No additional training data required.
- **Easy Integration** – Simple API for seamless use with existing machine learning workflows.

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
