# **SMERVisual**

SMERVisual is a Python package designed for explainable machine learning using the **SMER method**. It provides an intuitive interface for analyzing image classification models by highlighting key features and generating bounding boxes around important words in source images.

The package supports both **OpenAI API** models and **local language models**, offering flexibility in model selection.

---

## **Installation**
Install SMERVisual using pip:
```sh
pip install smer-visual
```

---

## **Features**

### **ImageClassifier**
The `ImageClassifier` class performs **logistic regression-based classification** on image datasets while computing **LIME** and **SMER** values for explainability. Key features include:

- Classification of image datasets using pre-trained language models
- Computation of **LIME and SMER scores** for each word in the dataset
- Visualization of **most important words** for explainability

### **BoundingBoxGenerator**
The `BoundingBoxGenerator` class enhances model interpretability by overlaying bounding boxes on images, highlighting **critical words** identified in classification. It:

- Uses **ImageClassifier output** to detect significant words
- Supports **vision models from OpenAI** and **open-source alternatives**
- Enables **zero-shot object detection** for explainable AI

---

## **Usage Example**
Here's a simple way to use **SMERVisual** for explainable image classification:

```python
from src import ImageClassifier, BoundingBoxGenerator

# Initialize and train the classifier
classifier = ImageClassifier(openai_model="openai", openai_key='123abc123')
classifier(data="path/to/dataset")
influential_words = classifier.get_top_words()

# Generate bounding boxes for top words
bbox_generator = BoundingBoxGenerator(data='path/to/dataset', top_words=influential_words,
                                      local_model_path='path/to/local/model')
bbox_generator()
```

---

## **Why Use SMERVisual?**

- **Explainable AI** – Provides insight into model decision-making
- **Model-Agnostic** – Compatible with OpenAI APIs and open-source models
- **Zero-Shot Detection** – No additional training data required
- **Easy Integration** – Simple API for seamless use with existing machine learning workflows

---

## **Contributing**
Contributions are welcome. If you’d like to contribute, follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

---

## **License**
SMERVisual is released under the **MIT License**.

---
