from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of input text.
    Args:
        text: Input string.
    Returns:
        Compound sentiment score (-1 to 1).
    """
    try:
        score = analyzer.polarity_scores(text)
        return score["compound"]
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return 0.0
