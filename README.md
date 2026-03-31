# NLP Project 2: Insurance Reviews Analysis

**ESILV A4 DIA6 — 2026**             
**Authors:** Leo WINTER & Alvaro SERERO

## Project Overview
This project focuses on the Natural Language Processing (NLP) analysis of insurance customer reviews. The dataset consists of reviews (avis) for various insurance products and insurers, originally in French and translated into English.

The goal is to extract insights, perform sentiment analysis, predict ratings, and build a dashboard for visualization.

## Project Structure
```
NLP_2/
├── data/                       # Contains 35 Excel files (avis_X_traduit.xlsx) with reviews
├── app/                        # Contains the file to run a streamlit app
├── notebooks/                  # Jupyter Notebooks for analysis and modeling
├── README.md                   # Project documentation
└── *.png                       # Generated plots (Rating distribution, Reviews per insurer, etc.)
```

## Dataset
The dataset is composed of **35 Excel files**, each representing reviews for different insurance segments. These are merged into a single DataFrame for analysis.

**Key Columns:**
| Column | Description |
| :--- | :--- |
| `note` | Star rating usually 1–5 (Target variable) |
| `avis` | Original review text in French |
| `avis_en` | Review translated to English |
| `avis_cor` | Corrected French text (planned) |
| `assureur` | Name of the insurance company |
| `produit` | Type of insurance product (auto, santé, habitation) |
| `date_publication` | Date when the review was posted |

## Roadmap
The project follows a structured pipeline:

1.  **Data Cleaning & EDA**
    *   Load and merge data.
    *   Visualizations: Rating distribution, reviews per insurer/product.
    *   Spelling correction and text normalization.

2.  **Summary & Translation**
    *   Generate summaries for long reviews.
    *   Ensure high-quality English translations.

3.  **Topic Modeling**
    *   Identify key themes (e.g., *price, customer service, claims*) using LDA or BERTopic.

4.  **Embeddings**
    *   Train Word2Vec or use pre-trained GloVe embeddings.
    *   Visualize semantic relationships.

5.  **Supervised Learning (Rating Prediction)**
    *   Model 1: TF-IDF + Classical ML (Logistic Regression / SVM).
    *   Model 2: Deep Learning with Embeddings (LSTM / BERT).

6.  **Deployment**
    *   Streamlit Dashboard for interactive analysis and prediction.

## Visualizations
The analysis generates several key insights:
*   **Rating Distribution:** Shows the balance of positive vs. negative reviews.
*   **Reviews per Insurer:** Identifies the most reviewed companies.
*   **Reviews per Product:** specific textual analysis per product type.

*(See generated `.png` files in the `visualizations` directory for examples)*

## Installation & Usage

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install pandas openpyxl matplotlib seaborn wordcloud textblob deep-translator tqdm nlpclean
    ```
3.  **Run the analysis**:
    Open `report_notebook.ipynb` in Jupyter Notebook.
