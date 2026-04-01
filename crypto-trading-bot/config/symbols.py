"""
Comprehensive symbol lists for crypto and stocks.

• CRYPTO_SYMBOLS  – top ~150 coins by market cap (yfinance format: XXX-USD)
• STOCK_SYMBOLS   – S&P 500 + popular mid-caps, ETFs, and meme stocks

These are the *available* universe.  The bot's active watchlist is a smaller
subset configured in config.py → CONFIG["crypto_watchlist"] / CONFIG["stock_watchlist"].
"""

# ── Top ~150 crypto by market cap (yfinance ticker format) ────────
CRYPTO_SYMBOLS: list[str] = [
    # Top 10
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "TRX-USD", "LINK-USD", "AVAX-USD",
    # 11-30
    "SHIB-USD", "DOT-USD", "LTC-USD", "XLM-USD", "ATOM-USD",
    "HBAR-USD", "FIL-USD", "ALGO-USD", "NEAR-USD", "ICP-USD",
    "VET-USD", "INJ-USD", "FET-USD", "AAVE-USD", "MKR-USD",
    "OP-USD", "ARB11841-USD", "CRO-USD", "MANA-USD", "SAND-USD",
    # 31-60
    "EOS-USD", "THETA-USD", "XTZ-USD", "IOTA-USD", "NEO-USD",
    "DASH-USD", "ZEC-USD", "BAT-USD", "ENJ-USD", "COMP-USD",
    "SNX-USD", "YFI-USD", "SUSHI-USD", "CRV-USD", "1INCH-USD",
    "GRT-USD", "LRC-USD", "AXS-USD", "CHZ-USD", "GALA-USD",
    "KAVA-USD", "ZIL-USD", "ICX-USD", "ONT-USD", "QTUM-USD",
    "ZRX-USD", "REN-USD", "ANKR-USD", "STORJ-USD", "SKL-USD",
    # 61-90
    "CELO-USD", "MASK-USD", "DYDX-USD",
    "FLR-USD", "JASMY-USD", "AMP-USD", "CKB-USD",
    "RSR-USD", "OCEAN-USD", "BAND-USD",
    "BAL-USD", "KNC-USD", "CELR-USD", "CTK-USD", "REEF-USD",
    "DENT-USD", "HOT-USD", "SC-USD", "IOTX-USD", "ONE-USD",
    "AUDIO-USD", "RLC-USD", "NKN-USD", "CTSI-USD", "RAD-USD",
    "REQ-USD", "POND-USD", "POLS-USD", "TVK-USD", "ALCX-USD",
    # 91-120
    "LPT-USD", "NMR-USD", "BNT-USD", "PERP-USD", "ALPHA-USD",
    "BADGER-USD", "MLN-USD", "FORTH-USD", "MIR-USD",
    "TRIBE-USD", "FEI-USD", "SPELL-USD",
    "T-USD", "BICO-USD",
    "GODS-USD", "IMX-USD", "LOKA-USD",
    "APE-USD", "GMT-USD", "OMG-USD",
    "COTI-USD", "AGLD-USD", "RARE-USD", "SUPER-USD",
    "HIGH-USD", "CLV-USD", "FLUX-USD", "XEC-USD",
    # 121-150
    "WAXP-USD", "HIVE-USD", "SXP-USD", "DGB-USD",
    "RVN-USD", "WAVES-USD", "XEM-USD", "GLMR-USD",
    "MOVR-USD", "MINA-USD", "KDA-USD", "CFX-USD",
    "ACH-USD", "BOBA-USD", "PEOPLE-USD",
    "LQTY-USD", "STMX-USD", "DUSK-USD", "TRAC-USD",
    "MDT-USD", "PHA-USD", "LIT-USD",
    "FIO-USD", "ERN-USD", "IDEX-USD",
    "DAR-USD", "FIDA-USD", "RNDR-USD",
]


# ── US stocks: S&P 500 + popular additions ────────────────────────
STOCK_SYMBOLS: list[str] = [
    # ─── Technology ───────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "NFLX", "INTC", "CSCO",
    "QCOM", "TXN", "NOW", "INTU", "AMAT", "LRCX", "MU", "ADI",
    "KLAC", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "ZS", "DDOG",
    "NET", "SNOW", "PLTR", "TEAM", "WDAY", "MRVL", "ON", "NXPI",
    "MPWR", "ANSS", "KEYS", "ZBRA", "SMCI", "DELL", "HPQ", "HPE",
    "ANET", "FICO", "IT", "EPAM", "AKAM", "FFIV", "JNPR", "NTAP",
    # ─── Finance ──────────────────────────────────────────────────
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
    "AXP", "C", "USB", "PNC", "TFC", "BK", "SPGI", "MCO", "ICE",
    "CME", "MSCI", "FIS", "FISV", "COF", "DFS", "MET", "PRU",
    "AIG", "AFL", "ALL", "TRV", "CB", "PGR", "HIG", "MMC", "AON",
    "AJG", "BRO", "WRB", "GL", "CINF", "RE", "BRK-B",
    # ─── Healthcare / Pharma / Bio ────────────────────────────────
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK",
    "BSX", "EW", "ZBH", "BDX", "CI", "ELV", "HCA", "HUM",
    "CNC", "MCK", "CAH", "ABC", "DXCM", "IDXX", "IQV", "A",
    "MTD", "WAT", "HOLX", "TECH", "ALGN", "RVTY", "BIO",
    # ─── Consumer / Retail ────────────────────────────────────────
    "WMT", "COST", "HD", "LOW", "TGT", "AMZN", "NKE", "SBUX",
    "MCD", "YUM", "CMG", "DPZ", "DARDEN", "KO", "PEP", "MNST",
    "STZ", "PM", "MO", "TAP", "SAM", "BF-B",
    "PG", "CL", "KMB", "CHD", "CLX", "SJM", "HRL", "MKC",
    "KHC", "GIS", "CAG", "CPB", "K", "HSY", "TSN", "SWK",
    "DG", "DLTR", "ROST", "TJX", "BBY", "LULU", "DECK", "CROX",
    "RH", "ETSY", "EBAY", "W", "CHWY",
    # ─── Energy ───────────────────────────────────────────────────
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
    "PXD", "OXY", "DVN", "HES", "FANG", "KMI", "WMB", "OKE",
    "HAL", "BKR", "TRGP", "CTRA",
    # ─── Industrial / Aerospace / Defense ─────────────────────────
    "CAT", "BA", "GE", "HON", "LMT", "RTX", "NOC", "GD",
    "UPS", "FDX", "UNP", "CSX", "NSC", "DE", "CMI", "EMR",
    "ETN", "ROK", "ITW", "PH", "PCAR", "WM", "RSG", "FAST",
    "GWW", "SHW", "APD", "ECL", "LIN", "PPG", "ALB", "VMC",
    "MLM", "DOV", "IR", "XYL", "A", "AME", "OTIS", "CARR",
    # ─── Telecom / Media / Communication ──────────────────────────
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "PARA", "WBD",
    "NWSA", "FOXA", "OMC", "IPG", "LYV", "TTWO", "EA", "ATVI",
    "RBLX", "U", "MTCH", "PINS", "SNAP", "SPOT",
    # ─── Real Estate (REITs) ──────────────────────────────────────
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL",
    "SPG", "AVB", "EQR", "MAA", "UDR", "CPT", "ESS", "VTR",
    "PEAK", "HST", "KIM", "REG", "FRT", "BXP", "SLG", "VNO",
    "ARE", "CBRE", "JLL",
    # ─── Utilities ────────────────────────────────────────────────
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "ED", "WEC", "ES", "AWK", "ATO", "CMS", "DTE", "FE",
    "EVRG", "PEG", "PPL", "AES", "CEG", "NRG", "VST",
    # ─── Materials ────────────────────────────────────────────────
    "NEM", "FCX", "GOLD", "NUE", "STLD", "CF", "MOS", "FMC",
    "DD", "DOW", "LYB", "CE", "EMN", "IP", "PKG", "WRK",
    "AVY", "SEE", "BLL", "AMCR",
    # ─── Popular non-S&P / meme / growth ──────────────────────────
    "RIVN", "LCID", "NIO", "XPEV", "LI", "SOFI", "HOOD", "COIN",
    "MARA", "RIOT", "MSTR", "ROKU", "SQ", "PYPL", "SHOP", "SE",
    "BABA", "JD", "PDD", "BILI", "FUTU", "GRAB", "CPNG", "DUOL",
    "ABNB", "UBER", "LYFT", "DASH", "OPEN", "CLOV", "WISH",
    "GME", "AMC", "BBBY", "BB", "NOK", "SPCE", "PLUG", "FCEL",
    "WKHS", "GOEV", "QS", "CHPT", "BLNK", "RUN", "ENPH", "SEDG",
    "FSLR", "CELH", "HIMS", "CAVA", "BIRK", "ARM", "IONQ", "RGTI",
    # ─── Major ETFs ───────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "ARKK",
    "ARKG", "ARKF", "ARKW", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLU", "XLY", "XLB", "XLRE", "GLD", "SLV", "USO",
    "TLT", "HYG", "LQD", "EEM", "EFA", "VWO", "VEA", "IEMG",
    "SOXL", "TQQQ", "SQQQ", "SPXU", "UVXY", "VXX",
]


def get_crypto_symbols() -> list[str]:
    """Return the full crypto universe."""
    return CRYPTO_SYMBOLS


def get_stock_symbols() -> list[str]:
    """Return the full stock universe."""
    return STOCK_SYMBOLS


def get_all_available_symbols() -> list[str]:
    """Return every tradeable symbol (crypto + stocks)."""
    return CRYPTO_SYMBOLS + STOCK_SYMBOLS
