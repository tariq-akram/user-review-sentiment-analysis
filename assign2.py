import pandas as pd
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')


def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")



def clean_data(df):
    # Drop rows with null values
    df.dropna(inplace=True)
    # Drop unnecessary columns, assuming we need only 'review' column
    df = df[['review']]
    return df


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text



def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for review in df['review']:
        ss = sid.polarity_scores(review)
        if ss['compound'] >= 0.05:
            sentiments.append('positive')
        elif ss['compound'] <= -0.05:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    df['sentiment'] = sentiments
    return df

def generate_summary(df):
    sentiment_counts = df['sentiment'].value_counts()
    summary_report = {
        'positive': sentiment_counts.get('positive', 0),
        'negative': sentiment_counts.get('negative', 0),
        'neutral': sentiment_counts.get('neutral', 0)
    }
    return summary_report

def main(file_path):
    df = load_dataset(file_path)
    df = clean_data(df)
    df['review'] = df['review'].apply(preprocess_text)
    df = analyze_sentiment(df)
    summary_report = generate_summary(df)

    print("Summary Report:")
    print(f"Positive reviews: {summary_report['positive']}")
    print(f"Negative reviews: {summary_report['negative']}")
    print(f"Neutral reviews: {summary_report['neutral']}")




# Provide the path to your CSV file
file_path = 'user_review.xls'
main(file_path)
