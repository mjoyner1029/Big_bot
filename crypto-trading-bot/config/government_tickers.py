"""
Company Name to Ticker Symbol Mapping
======================================
Maps company names to stock ticker symbols for matching government contracts
to tradeable stocks.

Focus on:
- Defense contractors (biggest gov spenders)  
- Healthcare/Medical suppliers
- Technology contractors
- Small/mid-cap companies (higher impact from contracts)
"""

# Defense & Aerospace
DEFENSE_CONTRACTORS = {
    # Large caps
    "LMT": "LOCKHEED MARTIN",
    "RTX": "RAYTHEON TECHNOLOGIES",
    "NOC": "NORTHROP GRUMMAN",
    "GD": "GENERAL DYNAMICS",
    "BA": "BOEING",
    "LHX": "L3HARRIS TECHNOLOGIES",
    "HII": "HUNTINGTON INGALLS INDUSTRIES",
    "TXT": "TEXTRON",
    "TDG": "TRANSDIGM GROUP",
    
    # Mid/Small caps (higher % impact from contracts)
    "KTOS": "KRATOS DEFENSE & SECURITY SOLUTIONS",
    "AVAV": "AEROVIRONMENT",
    "ATRO": "ASTRONICS",
    "CW": "CURTISS-WRIGHT",
    "MRCY": "MERCURY SYSTEMS",
    "AJRD": "AEROJET ROCKETDYNE",
    "HXL": "HEXCEL",
    "TGI": "TRIUMPH GROUP",
    "AAR": "AAR CORP",
    "ESLT": "ELBIT SYSTEMS",
    "VSI": "VERTEX STANDARD",
    "COHR": "COHERENT",
    "OSIS": "OSI SYSTEMS",
    "NPK": "NATIONAL PRESTO INDUSTRIES",
    "AIR": "AAR CORP",
}

# Medical & Healthcare
HEALTHCARE_CONTRACTORS = {
    # Large health insurers (Medicare/Medicaid contracts)
    "UNH": "UNITEDHEALTH GROUP",
    "CVS": "CVS HEALTH",
    "CI": "CIGNA",
    "HUM": "HUMANA",
    "ANTM": "ANTHEM",
    "CNC": "CENTENE",
    "MOH": "MOLINA HEALTHCARE",
    
    # Medical device & supplies
    "MDT": "MEDTRONIC",
    "ABT": "ABBOTT LABORATORIES",
    "BAX": "BAXTER INTERNATIONAL",
    "BDX": "BECTON DICKINSON",
    "SYK": "STRYKER",
    "ZBH": "ZIMMER BIOMET",
    "BSX": "BOSTON SCIENTIFIC",
    "EW": "EDWARDS LIFESCIENCES",
    "HOLX": "HOLOGIC",
    "ALGN": "ALIGN TECHNOLOGY",
    "ISRG": "INTUITIVE SURGICAL",
    "DXCM": "DEXCOM",
    "PODD": "INSULET",
    "ICUI": "ICU MEDICAL",
    "NVST": "ENVISTA HOLDINGS",
    
    # Pharmaceuticals (VA contracts)
    "PFE": "PFIZER",
    "JNJ": "JOHNSON & JOHNSON",
    "MRK": "MERCK",
    "LLY": "ELI LILLY",
    "ABBV": "ABBVIE",
    "BMY": "BRISTOL-MYERS SQUIBB",
    "GILD": "GILEAD SCIENCES",
    "AMGN": "AMGEN",
    "REGN": "REGENERON PHARMACEUTICALS",
}

# Technology Contractors
TECH_CONTRACTORS = {
    # Big tech (cloud/AI for government)
    "MSFT": "MICROSOFT",
    "GOOGL": "GOOGLE",
    "AMZN": "AMAZON",
    "ORCL": "ORACLE",
    "IBM": "IBM",
    "CSCO": "CISCO SYSTEMS",
    "HPE": "HEWLETT PACKARD ENTERPRISE",
    "DELL": "DELL TECHNOLOGIES",
    
    # Cybersecurity (gov is huge buyer)
    "PANW": "PALO ALTO NETWORKS",
    "CRWD": "CROWDSTRIKE",
    "FTNT": "FORTINET",
    "ZS": "ZSCALER",
    "OKTA": "OKTA",
    "CYBR": "CYBERARK SOFTWARE",
    "S": "SENTINELONE",
    "NET": "CLOUDFLARE",
    "TENB": "TENABLE HOLDINGS",
    "RPD": "RAPID7",
    
    # Data analytics / AI
    "PLTR": "PALANTIR TECHNOLOGIES",
    "SNOW": "SNOWFLAKE",
    "DDOG": "DATADOG",
    "SPLK": "SPLUNK",
    
    # IT Services
    "ACN": "ACCENTURE",
    "IBM": "IBM",
    "CTSH": "COGNIZANT",
    "LDOS": "LEIDOS HOLDINGS",
    "SAIC": "SCIENCE APPLICATIONS INTERNATIONAL",
    "CACI": "CACI INTERNATIONAL",
    "MAN": "MANTECH INTERNATIONAL",
    "BAH": "BOOZ ALLEN HAMILTON",
}

# Infrastructure & Construction
INFRASTRUCTURE_CONTRACTORS = {
    "CAT": "CATERPILLAR",
    "DE": "DEERE & COMPANY",
    "FLR": "FLUOR",
    "PWR": "QUANTA SERVICES",
    "EME": "EMCOR GROUP",
    "MTZ": "MASTEC",
    "ACM": "AECOM",
    "JEC": "JACOBS ENGINEERING",
    "KBR": "KBR",
}

# Energy & Utilities (DoE contracts)
ENERGY_CONTRACTORS = {
    "NEE": "NEXTERA ENERGY",
    "DUK": "DUKE ENERGY",
    "SO": "SOUTHERN COMPANY",
    "D": "DOMINION ENERGY",
    "AEP": "AMERICAN ELECTRIC POWER",
    "EXC": "EXELON",
    "XEL": "XCEL ENERGY",
    "SRE": "SEMPRA ENERGY",
    "PEG": "PUBLIC SERVICE ENTERPRISE GROUP",
}


def get_all_ticker_mappings() -> dict:
    """Combine all sector mappings into one dictionary."""
    combined = {}
    combined.update(DEFENSE_CONTRACTORS)
    combined.update(HEALTHCARE_CONTRACTORS)
    combined.update(TECH_CONTRACTORS)
    combined.update(INFRASTRUCTURE_CONTRACTORS)
    combined.update(ENERGY_CONTRACTORS)
    return combined


def get_ticker_for_company(company_name: str) -> str:
    """
    Find ticker symbol for a company name.
    
    Args:
        company_name: Company name from USAspending (e.g., "LOCKHEED MARTIN CORPORATION")
    
    Returns:
        Ticker symbol or empty string if not found
    """
    if not company_name:
        return ""
    
    # Clean up company name
    name = company_name.upper().strip()
    
    # Remove common suffixes
    for suffix in [" INC", " INCORPORATED", " CORP", " CORPORATION", " LLC", " LTD", " LIMITED"]:
        name = name.replace(suffix, "")
    
    name = name.strip()
    
    # Search all mappings
    all_mappings = get_all_ticker_mappings()
    
    for ticker, company in all_mappings.items():
        # Exact match
        if name == company:
            return ticker
        
        # Substring match (company name contains the mapping)
        if company in name or name in company:
            return ticker
    
    return ""


def get_high_impact_tickers() -> list:
    """
    Return list of tickers most likely to react strongly to contract news.
    Small/mid caps where a $50M contract is significant relative to market cap.
    """
    # Focus on smaller defense contractors and tech companies
    high_impact = [
        # Small defense (contracts = 5-20% of annual revenue)
        "KTOS", "AVAV", "ATRO", "CW", "MRCY", "AJRD",
        
        # Mid-cap defense
        "HII", "TXT", "HXL", "TGI",
        
        # Tech/cyber (government contracts signal legitimacy)
        "PLTR", "CRWD", "ZS", "S", "NET", "DDOG",
        
        # IT services (contract = immediate revenue)
        "SAIC", "CACI", "MAN", "LDOS",
        
        # Medical devices (VA/Medicare high margins)
        "ICUI", "PODD", "NVST", "HOLX",
    ]
    
    return high_impact


if __name__ == "__main__":
    # Test the mapping
    test_companies = [
        "LOCKHEED MARTIN CORPORATION",
        "PALANTIR TECHNOLOGIES INC",
        "KRATOS DEFENSE & SECURITY SOLUTIONS INC",
        "MICROSOFT CORPORATION",
    ]
    
    print("Testing company name to ticker mapping:")
    for company in test_companies:
        ticker = get_ticker_for_company(company)
        print(f"  {company} -> {ticker if ticker else 'NOT FOUND'}")
    
    print(f"\nHigh-impact tickers: {', '.join(get_high_impact_tickers())}")
