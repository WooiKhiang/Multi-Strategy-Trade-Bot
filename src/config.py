from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    gsheet_id: str = os.getenv("GSHEET_ID", "")
    google_service_account_json: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "./service_account.json")

    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_api_secret: str = os.getenv("ALPACA_API_SECRET", "")
    alpaca_paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    universe_static_file: str = os.getenv("UNIVERSE_STATIC_FILE", "./universe_static.csv")
    max_universe_tickers: int = int(os.getenv("MAX_UNIVERSE_TICKERS", "1200"))

    yf_batch_size: int = int(os.getenv("YF_BATCH_SIZE", "80"))
    yf_sleep_between_batch_sec: int = int(os.getenv("YF_SLEEP_BETWEEN_BATCH_SEC", "2"))

    candidate_ttl_minutes: int = int(os.getenv("CANDIDATE_TTL_MINUTES", "180"))

    touch_buffer_atr_mult: float = float(os.getenv("TOUCH_BUFFER_ATR_MULT", "0.10"))
    stop_buffer_atr_mult: float = float(os.getenv("STOP_BUFFER_ATR_MULT", "0.15"))
    confirm_break_buffer_atr_mult: float = float(os.getenv("CONFIRM_BREAK_BUFFER_ATR_MULT", "0.05"))

    risk_per_trade_usd: float = float(os.getenv("RISK_PER_TRADE_USD", "25"))
    take_profit_r_mult: float = float(os.getenv("TAKE_PROFIT_R_MULT", "2.0"))

    timezone: str = os.getenv("TIMEZONE", "America/New_York")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
