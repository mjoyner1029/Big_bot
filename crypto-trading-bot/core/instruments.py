"""Common instrument abstraction across stocks, ETFs and crypto.

Configurable universes with filters; ETF relationships (stock → sector ETF)
supporting relative-value research. Uses existing data providers only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Instrument:
    symbol: str
    asset_class: str                    # 'stock' | 'etf' | 'crypto'
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    market_cap: Optional[float] = None
    adv_usd: Optional[float] = None
    trading_hours: str = "24/7"
    supports_short: bool = False
    supports_fractional: bool = True
    data_capabilities: List[str] = field(default_factory=lambda: ["ohlcv"])
    etf_classification: Optional[str] = None   # broad|sector|factor|bond|...
    sector_etf: Optional[str] = None           # relationship: stock -> sector ETF


@dataclass
class UniverseFilters:
    min_price: float = 1.0
    min_market_cap: float = 0.0
    min_adv_usd: float = 1_000_000.0
    min_history_bars: int = 200
    exclude_otc: bool = True


# ── Built-in universes (extendable via config; no external services) ─────────

_CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "ADA-USD",
    "DOT-USD", "DOGE-USD", "AAVE-USD", "LTC-USD", "UNI-USD", "TAO-USD",
]

# Broad liquid US stock universe: sector -> (sector ETF, [symbols]).
# Curated large/mid-cap liquid names; extend or replace via universe file
# (load_universe_file) when a full-membership data provider is available.
_SECTOR_UNIVERSE: Dict[str, Dict] = {
    "semiconductor": {"etf": "SMH", "symbols": [
        "NVDA", "AMD", "MU", "AVGO", "INTC", "QCOM", "TXN", "ADI", "MRVL",
        "NXPI", "MCHP", "ON", "SWKS", "QRVO", "MPWR", "LRCX", "AMAT", "KLAC",
        "ASML", "TSM", "ARM", "SMCI", "TER", "ENTG",
    ]},
    "technology": {"etf": "XLK", "symbols": [
        "AAPL", "MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "IBM", "CSCO",
        "ACN", "SNOW", "PLTR", "PANW", "CRWD", "ZS", "FTNT", "DDOG", "NET",
        "MDB", "TEAM", "WDAY", "HPQ", "DELL", "ANET", "APH", "GLW", "SNPS",
        "CDNS", "ADSK", "SHOP", "SQ", "U", "TWLO", "OKTA", "DOCU", "ZM",
    ]},
    "communication": {"etf": "XLC", "symbols": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
        "EA", "TTWO", "RBLX", "SPOT", "PINS", "SNAP", "MTCH", "WBD", "PARA",
    ]},
    "consumer_discretionary": {"etf": "XLY", "symbols": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
        "ABNB", "MAR", "HLT", "CMG", "ORLY", "AZO", "ROST", "YUM", "DHI",
        "LEN", "PHM", "GM", "F", "RIVN", "LCID", "EBAY", "ETSY", "W", "RCL",
        "CCL", "NCLH", "LVS", "WYNN", "MGM", "DKNG",
    ]},
    "consumer_staples": {"etf": "XLP", "symbols": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "SYY", "KR", "TGT", "DG", "DLTR", "EL", "HSY", "K", "STZ",
        "TAP", "KHC", "CAG", "CPB", "TSN",
    ]},
    "financials": {"etf": "XLF", "symbols": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "V",
        "MA", "PYPL", "USB", "PNC", "TFC", "COF", "BK", "STT", "AIG", "MET",
        "PRU", "ALL", "TRV", "PGR", "CB", "MMC", "AON", "SPGI", "MCO", "ICE",
        "CME", "NDAQ", "KKR", "BX", "APO", "SOFI", "HOOD",
    ]},
    "crypto_equity": {"etf": "XLF", "symbols": [
        "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "GLXY", "CORZ",
    ]},
    "healthcare": {"etf": "XLV", "symbols": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "ISRG", "SYK",
        "BSX", "MDT", "EW", "REGN", "VRTX", "BIIB", "MRNA", "ZTS", "HCA",
        "MCK", "COR", "IDXX", "IQV", "A", "DXCM",
    ]},
    "energy": {"etf": "XLE", "symbols": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
        "PXD", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "DVN", "FANG",
        "APA", "TRGP", "CTRA", "MRO",
    ]},
    "industrials": {"etf": "XLI", "symbols": [
        "CAT", "DE", "UNP", "UPS", "FDX", "HON", "GE", "BA", "LMT", "RTX",
        "NOC", "GD", "MMM", "EMR", "ETN", "ITW", "PH", "CMI", "PCAR", "CSX",
        "NSC", "DAL", "UAL", "AAL", "LUV", "WM", "RSG", "URI", "FAST", "GWW",
        "TT", "CARR", "OTIS", "JCI", "ROK", "DOV", "AME", "XYL",
    ]},
    "materials": {"etf": "XLB", "symbols": [
        "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "DD", "PPG", "NUE",
        "STLD", "CLF", "AA", "VMC", "MLM", "ALB", "CF", "MOS", "IP",
    ]},
    "utilities": {"etf": "XLU", "symbols": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
        "WEC", "ES", "AWK", "DTE", "PPL", "FE", "AEE", "CMS", "CEG", "VST",
    ]},
    "real_estate": {"etf": "XLRE", "symbols": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR",
        "AVB", "EQR", "VTR", "SBAC", "WY", "IRM", "EXR", "MAA", "ARE",
    ]},
}

_STOCK_UNIVERSE: Dict[str, Dict] = {
    symbol: {"sector": sector, "sector_etf": meta["etf"]}
    for sector, meta in _SECTOR_UNIVERSE.items()
    for symbol in meta["symbols"]
}

_ETF_UNIVERSE: Dict[str, str] = {
    "SPY": "broad", "QQQ": "broad", "IWM": "broad", "DIA": "broad",
    "VTI": "broad", "VOO": "broad", "MDY": "broad", "RSP": "broad",
    "XLK": "sector", "SMH": "sector", "SOXX": "sector", "XLF": "sector",
    "XLE": "sector", "XLV": "sector", "XLY": "sector", "XLC": "sector",
    "XLI": "sector", "XLP": "sector", "XLU": "sector", "XLB": "sector",
    "XLRE": "sector", "XBI": "industry", "KRE": "industry", "XHB": "industry",
    "XOP": "industry", "ITA": "industry", "JETS": "industry", "TAN": "industry",
    "GLD": "commodity", "SLV": "commodity", "USO": "commodity", "UNG": "commodity",
    "DBA": "commodity", "CPER": "commodity",
    "TLT": "bond", "IEF": "bond", "SHY": "bond", "HYG": "bond", "LQD": "bond",
    "AGG": "bond", "TIP": "bond",
    "MTUM": "factor", "VLUE": "factor", "QUAL": "factor", "USMV": "factor",
    "SPLV": "factor", "IWF": "factor", "IWD": "factor",
    "TQQQ": "leveraged", "UPRO": "leveraged", "SQQQ": "inverse", "SH": "inverse",
    "EEM": "country", "EFA": "country", "FXI": "country", "EWJ": "country",
    "EWZ": "country", "INDA": "country", "EWY": "country", "EWT": "country",
    "BITO": "crypto", "IBIT": "crypto", "ETHA": "crypto",
}


class InstrumentUniverse:
    """Research universe across asset classes with configurable filters."""

    def __init__(self, filters: Optional[UniverseFilters] = None,
                 extra_stocks: Optional[Dict[str, Dict]] = None,
                 extra_etfs: Optional[Dict[str, str]] = None,
                 extra_crypto: Optional[List[str]] = None) -> None:
        self.filters = filters or UniverseFilters()
        self._stocks = {**_STOCK_UNIVERSE, **(extra_stocks or {})}
        self._etfs = {**_ETF_UNIVERSE, **(extra_etfs or {})}
        self._crypto = list(dict.fromkeys(_CRYPTO_SYMBOLS + (extra_crypto or [])))

    def crypto(self) -> List[Instrument]:
        return [Instrument(symbol=s, asset_class="crypto", exchange="coinbase",
                           trading_hours="24/7", supports_short=False,
                           data_capabilities=["ohlcv"])
                for s in self._crypto]

    def stocks(self, include_delisted: bool = False) -> List[Instrument]:
        return [Instrument(symbol=s, asset_class="stock", exchange="us_equity",
                           sector=meta.get("sector"),
                           industry=meta.get("industry"),
                           sector_etf=meta.get("sector_etf"),
                           market_cap=meta.get("market_cap"),
                           trading_hours="09:30-16:00 ET",
                           supports_short=True,
                           data_capabilities=["ohlcv"])
                for s, meta in self._stocks.items()
                if include_delisted or meta.get("active", True)]

    def etfs(self) -> List[Instrument]:
        return [Instrument(symbol=s, asset_class="etf", exchange="us_equity",
                           etf_classification=cls,
                           trading_hours="09:30-16:00 ET",
                           supports_short=True,
                           data_capabilities=["ohlcv"])
                for s, cls in self._etfs.items()]

    def all_instruments(self, asset_classes: Optional[List[str]] = None) -> List[Instrument]:
        out: List[Instrument] = []
        classes = asset_classes or ["crypto", "stock", "etf"]
        if "crypto" in classes:
            out.extend(self.crypto())
        if "stock" in classes:
            out.extend(self.stocks())
        if "etf" in classes:
            out.extend(self.etfs())
        return out

    def sector_etf_for(self, symbol: str) -> Optional[str]:
        meta = self._stocks.get(symbol)
        return meta.get("sector_etf") if meta else None

    def symbols_in_sector(self, sector: str) -> List[str]:
        return [s for s, meta in self._stocks.items()
                if meta.get("sector") == sector and meta.get("active", True)]

    def sectors(self) -> List[str]:
        return sorted({meta.get("sector") for meta in self._stocks.values()
                       if meta.get("sector")})

    def etfs_by_classification(self, classification: str) -> List[str]:
        return [s for s, cls in self._etfs.items() if cls == classification]

    def get(self, symbol: str) -> Optional[Instrument]:
        for inst in self.all_instruments():
            if inst.symbol == symbol:
                return inst
        return None

    def load_universe_file(self, path: str) -> int:
        """Merge an external universe file (CSV or JSON).

        CSV columns: symbol, asset_class, sector[, industry, sector_etf,
        market_cap, active, listed_from, listed_to]. Supports historical
        membership and delisted names via active/listed_* when the data
        provider permits. Returns number of instruments loaded.
        """
        import csv
        import json as _json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        rows: List[Dict] = []
        if p.suffix.lower() == ".json":
            rows = _json.loads(p.read_text())
        else:
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
        count = 0
        for row in rows:
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            asset_class = (row.get("asset_class") or "stock").lower()
            active = str(row.get("active", "true")).lower() not in ("false", "0", "no")
            if asset_class == "etf":
                self._etfs[symbol] = row.get("classification") or "unknown"
            elif asset_class == "crypto":
                if symbol not in self._crypto:
                    self._crypto.append(symbol)
            else:
                self._stocks[symbol] = {
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "sector_etf": row.get("sector_etf"),
                    "market_cap": float(row["market_cap"]) if row.get("market_cap") else None,
                    "active": active,
                    "listed_from": row.get("listed_from"),
                    "listed_to": row.get("listed_to"),
                }
            count += 1
        logger.info(f"InstrumentUniverse: loaded {count} instruments from {path}")
        return count


class UniverseResolver:
    """Resolves alpha universe tokens into concrete symbol lists.

    Tokens:
        "NVDA" / "BTC-USD"          explicit symbol
        "sector:semiconductor"      all active stocks in that sector
        "industry:biotech"          all stocks in that industry
        "class:crypto|stock|etf"    whole asset class
        "etf:sector|broad|bond..."  ETFs by classification
        "*"                         dynamic scanner top-N (caller-supplied)

    An alpha targeting US semiconductors therefore triggers a semiconductor-
    universe scan instead of whatever a generic top-10 list happens to hold.
    """

    def __init__(self, universe: Optional[InstrumentUniverse] = None,
                 max_symbols_per_alpha: int = 50) -> None:
        self.universe = universe or InstrumentUniverse()
        self.max_symbols_per_alpha = max_symbols_per_alpha

    def resolve(self, tokens: List[str],
                scanner_symbols: Optional[List[str]] = None) -> List[str]:
        out: List[str] = []
        for token in tokens or []:
            t = str(token).strip()
            if not t:
                continue
            if t == "*":
                out.extend(scanner_symbols or [])
            elif t.lower().startswith("sector:"):
                out.extend(self.universe.symbols_in_sector(t.split(":", 1)[1].lower()))
            elif t.lower().startswith("industry:"):
                industry = t.split(":", 1)[1].lower()
                out.extend(s for s, m in self.universe._stocks.items()
                           if (m.get("industry") or "").lower() == industry
                           and m.get("active", True))
            elif t.lower().startswith("class:"):
                cls = t.split(":", 1)[1].lower()
                out.extend(i.symbol for i in self.universe.all_instruments([cls]))
            elif t.lower().startswith("etf:"):
                out.extend(self.universe.etfs_by_classification(t.split(":", 1)[1].lower()))
            else:
                out.append(t.upper() if "-" not in t else t)
        # de-dupe preserving order, bounded
        seen = set()
        resolved = []
        for s in out:
            if s not in seen:
                seen.add(s)
                resolved.append(s)
        if len(resolved) > self.max_symbols_per_alpha:
            resolved = resolved[: self.max_symbols_per_alpha]
        return resolved
