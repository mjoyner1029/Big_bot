import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "capital": 500,
    "risk_per_trade": 10,
    "exchange": "coinbase",
    "use_paper_trading": True,
    "coinbase_api_key": os.getenv("COINBASE_API_KEY"),
    "coinbase_api_secret": os.getenv("COINBASE_API_SECRET"),
    "confidence_threshold": 0.6
}
