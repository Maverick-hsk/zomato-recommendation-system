# Zomato Content-Based Restaurant Recommendation System
[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://zomato-recommendation-system-9jjf8y7uzbgmxkasuy7izy.streamlit.app/)
A content-based recommendation engine for Zomato restaurants using **TF-IDF vectorization** and **Cosine Similarity**. Built on the full Kaggle dataset of **9,551 restaurants** with an interactive Streamlit UI and data visualizations.

## Features

- **Full Kaggle dataset** — 9,551 restaurants across global cities
- **Streamlit web app** — interactive recommendations with filters
- **Visualizations** — city/cuisine distributions, rating charts, similarity heatmaps, TF-IDF terms
- **Data preprocessing** — cleaning, normalization, and NLTK-based tokenization
- **Feature engineering** — composite restaurant profile text from metadata + reviews
- **TF-IDF + Cosine Similarity** — content-based similarity matching
- **Precision@K evaluation** — recall, hit rate, and MAP metrics
- **A/B testing** — compare TF-IDF hyperparameter configurations

## Project Structure

```
zomato-recommendation-system/
├── app.py                   # Streamlit web UI
├── data/
│   └── zomato.csv           # Full Kaggle dataset (9,551 rows)
├── notebooks/
├── scripts/
│   ├── download_dataset.py  # Auto-download full dataset
│   └── generate_sample_data.py
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── recommender.py
│   ├── evaluation.py
│   ├── ab_testing.py
│   ├── visualizations.py    # Plotly charts
│   └── data_utils.py
├── main.py
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
cd zomato-recommendation-system
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt   # includes notebooks; use requirements.txt for app-only
```

### 2. Download the full dataset (auto on first run)

```bash
python scripts/download_dataset.py
```

This fetches the official Kaggle schema dataset (~9,551 restaurants) into `data/zomato.csv`.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Notes for cloud deploy**

- `requirements.txt` is trimmed for production (no Jupyter).
- On first launch the app downloads the full dataset (~9.5k restaurants) and trains TF-IDF; expect a 1–2 minute cold start on the free tier.
- `data/sample_zomato.csv` is bundled as a fallback if the download fails.
- No secrets or API keys are required.

### 4. CLI usage

```bash
# Demo recommendations on full dataset
python main.py

# Similar to a specific restaurant
python main.py --restaurant "Punjabi By Nature"

# Match user preferences
python main.py --cuisines "North Indian, Mughlai" --city "New Delhi" --price-range 2

# Evaluate precision metrics
python main.py --evaluate

# A/B test TF-IDF configurations
python main.py --ab-test
```

## Streamlit App

| Tab | Contents |
|-----|----------|
| **Recommendations** | Similar-restaurant or preference-based search with similarity charts and heatmaps |
| **Visualizations** | City/cuisine bar charts, rating histogram, price pie chart, rating vs price box plot, TF-IDF terms |
| **Evaluation & A/B Test** | Precision@K metrics and hyperparameter comparison |

## Visualizations

- Top cities and cuisines by restaurant count
- Rating distribution and price-range breakdown
- Rating vs price box plot
- Recommendation similarity score bar chart
- Cosine similarity heatmap for top matches
- TF-IDF distinctive term analysis
- Evaluation metrics and A/B test comparison charts

## How It Works

### Profile Construction

Each restaurant is represented as a text document combining cuisine (2x weight), location, price range, review signals, and name.

### TF-IDF + Cosine Similarity

```python
TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
```

### Evaluation

Pseudo ground-truth: same city + overlapping cuisines + rating ≥ 3.5 = relevant.

## Tech Stack

- Python 3.10+
- scikit-learn, pandas, NumPy, NLTK
- Streamlit, Plotly

## License

MIT — for educational and portfolio use.
