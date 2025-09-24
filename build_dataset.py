import os
import re
import pandas as pd
import datetime
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ✅ Fallback clean_text function (in case utils.preprocess not available)
try:
    from utils.preprocess import clean_text
except ImportError:
    def clean_text(text: str) -> str:
        """Basic fallback cleaner: lowercase + alphanumeric filter."""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return ' '.join(text.split())

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = r"D:\Major_Project\data\transcripts"
OUTPUT_FILE = r"D:\Major_Project\data\processed\master_sentiment.csv"

START_DATE = datetime.date(2016, 1, 1)
END_DATE   = datetime.date(2020, 12, 31)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# SENTIMENT MODELS
# -----------------------------
vader_analyzer = SentimentIntensityAnalyzer()

USE_FINBERT = True   # enable FinBERT if available
if USE_FINBERT:
    try:
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    except Exception as e:
        logging.error(f"Failed to load FinBERT: {e}")
        USE_FINBERT = False

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_vader_sentiment(text: str):
    """Return compound score (-1..1) and label."""
    if not text.strip():
        return 0.0, "neutral"
    scores = vader_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound > 0.05:
        return compound, "positive"
    elif compound < -0.05:
        return compound, "negative"
    else:
        return compound, "neutral"

def get_finbert_sentiment(text: str, max_length=512):
    """Return (polarity, probs dict, tone label)."""
    if not text.strip() or not USE_FINBERT:
        return 0.0, {"neutral":0.0, "positive":0.0, "negative":0.0}, "neutral"
    
    inputs = finbert_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    
    sentiment_probs = {
        "neutral": float(probs[0]),
        "positive": float(probs[1]),
        "negative": float(probs[2])
    }
    polarity = sentiment_probs["positive"] - sentiment_probs["negative"]
    
    # Determine overall tone
    tone = max(sentiment_probs, key=sentiment_probs.get)
    return polarity, sentiment_probs, tone

def parse_filename(filename: str):
    """
    Extracts date and ticker from filename like '2016-Apr-26-AAPL.txt'.
    Supports tickers like 'BRK-A'.
    """
    base = filename.replace(".txt","")
    match = re.match(r"^(\d{4})-([A-Za-z]{3})-(\d{2})-(.+)$", base)
    if not match:
        return None, None
    year, month, day, ticker = match.groups()
    try:
        call_date = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%b-%d").date()
        return call_date, ticker
    except Exception:
        return None, None

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def build_master_dataset():
    records = []

    for company_folder in os.listdir(BASE_DIR):
        company_path = os.path.join(BASE_DIR, company_folder)
        if not os.path.isdir(company_path):
            continue

        txt_files = [f for f in os.listdir(company_path) if f.endswith(".txt")]
        for filename in tqdm(txt_files, desc=f"Processing {company_folder}"):
            filepath = os.path.join(company_path, filename)
            call_date, ticker = parse_filename(filename)
            if ticker is None: 
                continue

            # ✅ restrict to 2016–2020
            if call_date < START_DATE or call_date > END_DATE:
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                logging.error(f"Error reading {filepath}: {e}")
                continue

            # Sentiment analysis
            vader_score, vader_label = get_vader_sentiment(text)
            cleaned = clean_text(text)
            finbert_score, finbert_probs, finbert_tone = get_finbert_sentiment(cleaned)

            records.append({
                "company_folder": company_folder,
                "ticker": ticker,
                "date": call_date,
                "filename": filename,
                "vader_sentiment": vader_score,
                "vader_label": vader_label,
                "finbert_sentiment": finbert_score,
                "finbert_probs": finbert_probs,
                "finbert_tone": finbert_tone,
                "cleaned_text_preview": cleaned[:300],
                "raw_text_path": filepath
            })

    df = pd.DataFrame(records)
    if df.empty:
        logging.warning("❌ No valid records found in date range.")
        return df

    df = df.sort_values(["ticker","date"])
    df["vader_sentiment_ffill"] = df.groupby("ticker")["vader_sentiment"].ffill().fillna(0.0)
    
    if USE_FINBERT:
        df["finbert_sentiment_ffill"] = df.groupby("ticker")["finbert_sentiment"].ffill().fillna(0.0)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"✅ Master dataset saved at {OUTPUT_FILE}")

    return df

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    df = build_master_dataset()
    if not df.empty:
        print(df.head())
        print(f"\n✅ Final dataset shape: {df.shape}")
    else:
        print("No data produced. Check transcripts and dates.")