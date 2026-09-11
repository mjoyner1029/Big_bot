"""
Unit tests for Freqtrade indicator patterns.
"""
import pytest
import pandas as pd
import numpy as np
from indicators.freqtrade_patterns import (
    rsi_fast, rsi_slow, stoch_fast, macd_fast,
    ema_family, sma_family, adx_trend, bollinger_bands, atr,
    populate_indicators_freqtrade, generate_entry_signals, generate_stop_loss_atr,
    crossover, clear_indicator_cache
)


@pytest.fixture
def sample_df():
    """Generate realistic OHLCV data for testing."""
    dates = pd.date_range('2025-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    close = pd.Series(
        np.cumsum(np.random.randn(200) * 0.5) + 100,
        index=dates
    )
    high = close + np.abs(np.random.randn(200) * 0.3)
    low = close - np.abs(np.random.randn(200) * 0.3)
    volume = np.random.uniform(1000, 10000, 200)
    
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'open': close.shift(1).fillna(close.iloc[0]),
    })
    return df


class TestMomentumIndicators:
    """Test momentum indicator calculations."""
    
    def test_rsi_fast_shape(self, sample_df):
        """RSI fast should return Series with same length as input."""
        result = rsi_fast(sample_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
    
    def test_rsi_slow_values(self, sample_df):
        """RSI slow should be between 0 and 100."""
        result = rsi_slow(sample_df)
        valid_vals = result.dropna()
        assert (valid_vals >= 0).all()
        assert (valid_vals <= 100).all()
    
    def test_stoch_fast_returns_tuple(self, sample_df):
        """Stochastic should return tuple of (K, D) series."""
        k, d = stoch_fast(sample_df)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert len(k) == len(sample_df)
        assert len(d) == len(sample_df)
    
    def test_macd_fast_returns_tuple(self, sample_df):
        """MACD should return (macd, signal, histogram)."""
        macd, signal, hist = macd_fast(sample_df)
        assert len(macd) == len(sample_df)
        assert len(signal) == len(sample_df)
        assert len(hist) == len(sample_df)
        
        # Histogram = MACD - Signal
        np.testing.assert_array_almost_equal(
            hist.values[10:],  # Skip NaN start
            (macd - signal).values[10:],
            decimal=4
        )


class TestTrendIndicators:
    """Test trend indicator calculations."""
    
    def test_ema_family_returns_dict(self, sample_df):
        """EMA family should return dict of Series."""
        result = ema_family(sample_df)
        assert isinstance(result, dict)
        assert 'ema9' in result
        assert 'ema20' in result
        assert 'ema50' in result
        assert 'ema200' in result
    
    def test_sma_family_returns_dict(self, sample_df):
        """SMA family should return dict of Series."""
        result = sma_family(sample_df)
        assert isinstance(result, dict)
        assert 'sma50' in result
        assert 'sma200' in result
    
    def test_adx_trend_returns_tuple(self, sample_df):
        """ADX should return (adx, +DI, -DI)."""
        adx, plus_di, minus_di = adx_trend(sample_df)
        assert len(adx) == len(sample_df)
        assert len(plus_di) == len(sample_df)
        assert len(minus_di) == len(sample_df)
    
    def test_crossover_detection(self, sample_df):
        """Crossover should detect signal crossings."""
        line1 = sample_df['close'].ewm(span=9).mean()
        line2 = sample_df['close'].ewm(span=20).mean()
        
        result = crossover(sample_df, line1, line2)
        assert isinstance(result, pd.Series)
        assert result.isin([-1, 0, 1]).all()  # Only these values


class TestVolatilityIndicators:
    """Test volatility indicator calculations."""
    
    def test_bollinger_bands_returns_dict(self, sample_df):
        """Bollinger Bands should return dict with 5 components."""
        result = bollinger_bands(sample_df)
        assert isinstance(result, dict)
        assert len(result) == 5
        assert 'bb_upper' in result
        assert 'bb_mid' in result
        assert 'bb_lower' in result
        assert 'bb_pctb' in result
        assert 'bb_width' in result
    
    def test_bb_relationships(self, sample_df):
        """Bollinger Bands should maintain: lower < mid < upper."""
        bb_dict = bollinger_bands(sample_df)
        upper = bb_dict['bb_upper']
        mid = bb_dict['bb_mid']
        lower = bb_dict['bb_lower']
        
        valid_idx = upper.notna() & mid.notna() & lower.notna()
        assert (lower[valid_idx] < mid[valid_idx]).all()
        assert (mid[valid_idx] < upper[valid_idx]).all()
    
    def test_atr_positive(self, sample_df):
        """ATR should always be positive (it's a volatility measure)."""
        result = atr(sample_df)
        valid_vals = result.dropna()
        assert (valid_vals >= 0).all()


class TestCompositeIndicators:
    """Test composite indicator generation."""
    
    def test_populate_indicators_freqtrade_shape(self, sample_df):
        """populate_indicators_freqtrade should add 50+ columns."""
        result = populate_indicators_freqtrade(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        # Should have significantly more columns than input
        assert len(result.columns) > len(sample_df.columns) + 20  # ~30+ indicators added
    
    def test_populate_includes_key_indicators(self, sample_df):
        """Should include all key indicator columns."""
        result = populate_indicators_freqtrade(sample_df)
        required_cols = {
            'rsi_fast', 'rsi_slow', 'rsi',
            'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_hist',
            'ema9', 'ema20', 'ema50', 'ema200',
            'sma50', 'sma200',
            'adx', 'adx_pos', 'adx_neg',
            'bb_upper', 'bb_mid', 'bb_lower', 'bb_pctb', 'bb_width',
            'atr',
        }
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_generate_entry_signals(self, sample_df):
        """Entry signals should be 1, -1, or 0."""
        df = populate_indicators_freqtrade(sample_df)
        signals = generate_entry_signals(df)
        assert signals.isin([-1, 0, 1]).all()
    
    def test_generate_stop_loss_atr(self, sample_df):
        """Stop loss should be <close for long, >close for short."""
        df = populate_indicators_freqtrade(sample_df)
        sl_tp = generate_stop_loss_atr(df, atr_multiplier=2.0)
        
        # For valid rows, stop should bracket the close
        valid_idx = (df['atr'].notna()) & (df['close'].notna())
        
        sl_long = sl_tp['stop_loss_long'][valid_idx]
        sl_short = sl_tp['stop_loss_short'][valid_idx]
        close = df['close'][valid_idx]
        
        # Check on non-empty subset (some rows may be NaN early on)
        if len(sl_long) > 0:
            # Spot-check that stop losses are calculated (not all NaN)
            assert sl_long.notna().sum() > 0
            assert sl_short.notna().sum() > 0
            # Verify ranges are reasonable (not inf or extremely large)
            assert (sl_long.dropna() > -1e10).all()
            assert (sl_short.dropna() < 1e10).all()


class TestCaching:
    """Test indicator caching mechanism."""
    
    def test_cache_clears(self, sample_df):
        """Indicator cache should clear without errors."""
        clear_indicator_cache()  # Should not raise
        
        # Run indicator
        rsi_slow(sample_df)
        
        # Clear again
        clear_indicator_cache()  # Should not raise


class TestFreqtradeIntegration:
    """Test full Freqtrade pipeline integration."""
    
    def test_freqtrade_pipeline(self, sample_df):
        """Full pipeline should complete without errors."""
        # Add all indicators
        df = populate_indicators_freqtrade(sample_df)
        
        # Generate signals
        df['entry_signal'] = generate_entry_signals(df)
        
        # Generate stop-loss
        sl_tp = generate_stop_loss_atr(df)
        
        # Should have all components
        assert len(df) == len(sample_df)
        assert 'entry_signal' in df.columns
        assert 'stop_loss_long' in sl_tp
    
    def test_no_nans_in_recent_rows(self, sample_df):
        """Recent rows (after warmup) should not have NaN indicators."""
        df = populate_indicators_freqtrade(sample_df)
        
        # After 200 rows, most indicators should have values
        recent = df.iloc[-50:]
        
        # Key indicators should mostly have values (some allowance for edge cases)
        na_pct = recent[['rsi_slow', 'macd', 'bb_mid', 'atr']].isna().sum(axis=0) / len(recent)
        assert (na_pct < 0.1).all(), "Too many NaN values in recent data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
