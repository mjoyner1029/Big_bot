#!/usr/bin/env python3
"""Check API key configuration and test connectivity."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import CONFIG

def check_api_keys():
    """Check if API keys are configured."""
    print('=' * 70)
    print('API KEY CONFIGURATION STATUS')
    print('=' * 70)
    print()
    
    # Check Coinbase
    cb_key = CONFIG.get('coinbase_api_key', '')
    cb_secret = CONFIG.get('coinbase_api_secret', '')
    cb_pass = CONFIG.get('coinbase_passphrase', '')
    print(f'🔑 Coinbase API:')
    print(f'   Key:        {check_key(cb_key)}')
    print(f'   Secret:     {check_key(cb_secret)}')
    print(f'   Passphrase: {check_key(cb_pass)}')
    print()
    
    # Check Alpaca
    ap_key = CONFIG.get('alpaca_api_key', '')
    ap_secret = CONFIG.get('alpaca_api_secret', '')
    ap_url = CONFIG.get('alpaca_base_url', '')
    print(f'🔑 Alpaca API:')
    print(f'   Key:        {check_key(ap_key)}')
    print(f'   Secret:     {check_key(ap_secret)}')
    print(f'   Base URL:   {ap_url if ap_url else "❌ NOT SET"}')
    print()
    
    # Check Anthropic (Claude)
    anth_key = CONFIG.get('anthropic_api_key', '')
    anth_model = CONFIG.get('anthropic_model', '')
    use_llm = CONFIG.get('use_llm', False)
    print(f'🔑 Anthropic (Claude) API:')
    print(f'   Key:        {check_key(anth_key)}')
    print(f'   Model:      {anth_model if anth_model else "N/A"}')
    print(f'   Enabled:    {"✅ YES" if use_llm else "❌ NO (Disabled in config)"}')
    print()
    
    # Check News API
    news_key = CONFIG.get('news_api_key', '')
    print(f'🔑 News API:')
    print(f'   Key:        {check_key(news_key)}')
    print()
    
    # Check Trading Settings
    capital = CONFIG.get('capital', 0)
    paper = CONFIG.get('use_paper_trading', True)
    print(f'💰 Trading Settings:')
    print(f'   Capital:      ${capital:,.2f}')
    print(f'   Paper Mode:   {"✅ ENABLED (Safe)" if paper else "⚠️  LIVE TRADING (Real Money)"}')
    print()
    
    # Check Gov Contracts
    gov_enabled = CONFIG.get('enable_gov_contracts', False)
    min_contract = CONFIG.get('min_contract_amount', 0)
    max_contract = CONFIG.get('max_contract_amount', 0)
    print(f'🏛️  Government Contracts (USAspending.gov):')
    print(f'   Enabled:      {"✅ YES" if gov_enabled else "❌ NO"}')
    print(f'   Min Amount:   ${min_contract:,}')
    print(f'   Max Amount:   ${max_contract:,}')
    print(f'   Note:         No API key required - public API')
    print()
    
    # Check .env file
    print(f'📄 Environment File:')
    env_locations = [
        '../.env',
        '.env',
        '/Users/mjoyner/Data-AI/Big_bot/.env',
    ]
    env_found = False
    for env_path in env_locations:
        if os.path.exists(env_path):
            print(f'   ✅ Found: {os.path.abspath(env_path)}')
            env_found = True
            break
    if not env_found:
        print(f'   ❌ No .env file found in expected locations')
        print(f'      Create .env file with your API keys')
    print()
    
    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    
    keys_configured = []
    keys_missing = []
    
    if cb_key and cb_secret and cb_pass:
        keys_configured.append('✅ Coinbase (all credentials)')
    else:
        keys_missing.append('❌ Coinbase (missing credentials)')
    
    if ap_key and ap_secret:
        keys_configured.append('✅ Alpaca (key & secret)')
    else:
        keys_missing.append('❌ Alpaca (missing credentials)')
    
    if anth_key:
        if use_llm:
            keys_configured.append('✅ Anthropic (key set, LLM enabled)')
        else:
            keys_configured.append('⚠️  Anthropic (key set but LLM disabled)')
    else:
        keys_missing.append('❌ Anthropic (no key)')
    
    if news_key:
        keys_configured.append('✅ News API')
    else:
        keys_missing.append('❌ News API')
    
    keys_configured.append('✅ USAspending.gov (no key required)')
    
    print('\nConfigured:')
    for item in keys_configured:
        print(f'  {item}')
    
    if keys_missing:
        print('\nMissing:')
        for item in keys_missing:
            print(f'  {item}')
    
    print()
    print('=' * 70)


def check_key(key_value):
    """Format key status for display."""
    if not key_value:
        return '❌ NOT SET'
    return f'✅ SET ({len(key_value)} chars)'


def test_api_connectivity():
    """Test actual API connectivity."""
    print()
    print('=' * 70)
    print('API CONNECTIVITY TESTS')
    print('=' * 70)
    print()
    
    # Test Alpaca
    print('Testing Alpaca API...')
    try:
        ap_key = CONFIG.get('alpaca_api_key', '')
        ap_secret = CONFIG.get('alpaca_api_secret', '')
        ap_url = CONFIG.get('alpaca_base_url', '')
        
        if ap_key and ap_secret:
            import requests
            headers = {
                'APCA-API-KEY-ID': ap_key,
                'APCA-API-SECRET-KEY': ap_secret
            }
            response = requests.get(f'{ap_url}/v2/account', headers=headers, timeout=5)
            if response.status_code == 200:
                account = response.json()
                print(f'  ✅ Alpaca: Connected')
                print(f'     Account: {account.get("account_number", "N/A")}')
                print(f'     Cash: ${float(account.get("cash", 0)):,.2f}')
                print(f'     Buying Power: ${float(account.get("buying_power", 0)):,.2f}')
            elif response.status_code == 401:
                print(f'  ❌ Alpaca: Authentication failed (invalid credentials)')
            else:
                print(f'  ⚠️  Alpaca: Error {response.status_code}')
        else:
            print(f'  ⏭️  Alpaca: Skipped (credentials not set)')
    except Exception as e:
        print(f'  ❌ Alpaca: Error - {str(e)[:60]}')
    print()
    
    # Test Anthropic
    print('Testing Anthropic API...')
    try:
        anth_key = CONFIG.get('anthropic_api_key', '')
        if anth_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anth_key)
            # Simple test with minimal token usage
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f'  ✅ Anthropic: Connected')
            print(f'     Model: {response.model}')
        else:
            print(f'  ⏭️  Anthropic: Skipped (key not set)')
    except Exception as e:
        print(f'  ❌ Anthropic: Error - {str(e)[:60]}')
    print()
    
    # Test News API
    print('Testing News API...')
    try:
        news_key = CONFIG.get('news_api_key', '')
        if news_key:
            import requests
            url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={news_key}&pageSize=1'
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f'  ✅ News API: Connected')
                print(f'     Total Results: {data.get("totalResults", 0)}')
            elif response.status_code == 401:
                print(f'  ❌ News API: Authentication failed (invalid key)')
            else:
                print(f'  ⚠️  News API: Error {response.status_code}')
        else:
            print(f'  ⏭️  News API: Skipped (key not set)')
    except Exception as e:
        print(f'  ❌ News API: Error - {str(e)[:60]}')
    print()
    
    # Test USAspending.gov
    print('Testing USAspending.gov API...')
    try:
        import requests
        url = 'https://api.usaspending.gov/api/v2/search/spending_by_award/'
        payload = {
            "filters": {
                "time_period": [{"start_date": "2026-04-20", "end_date": "2026-04-27"}],
                "award_type_codes": ["A", "B", "C", "D"]
            },
            "fields": ["Award ID"],
            "page": 1,
            "limit": 1
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f'  ✅ USAspending.gov: Connected (public API)')
            print(f'     Results: {len(data.get("results", []))} contracts found')
        else:
            print(f'  ⚠️  USAspending.gov: Error {response.status_code}')
    except Exception as e:
        print(f'  ❌ USAspending.gov: Error - {str(e)[:60]}')
    print()
    
    print('=' * 70)


if __name__ == '__main__':
    import sys
    
    check_api_keys()
    
    # Check if --test flag is passed
    if '--test' in sys.argv or len(sys.argv) > 1 and sys.argv[1] in ('test', 'connectivity', '-t'):
        test_api_connectivity()
    else:
        # Ask if user wants to test connectivity
        print('\nRun with --test flag to test API connectivity')
        print('Example: python check_api_keys.py --test')
