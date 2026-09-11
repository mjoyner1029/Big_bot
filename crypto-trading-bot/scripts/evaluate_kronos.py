#!/usr/bin/env python3
"""
Evaluate Kronos directional accuracy on historical data.

Tests whether Kronos predictions are useful for trading by measuring
how often it correctly predicts price direction.
"""
import sys
sys.path.insert(0, '/Users/asad/Code/Big_bot/crypto-trading-bot')

import pandas as pd
from datetime import datetime, timedelta
from data.fetcher import fetch_latest_market_data
from core.kronos_predictor import is_kronos_available, BigBotKronosPredictor

def evaluate_symbol(predictor, symbol, lookback=300, pred_len=12, test_windows=10):
    """Evaluate Kronos on one symbol.
    
    Args:
        predictor: BigBotKronosPredictor instance
        symbol: Symbol to test
        lookback: Historical candles to use (reduced to 300 due to data limits)
        pred_len: Candles to predict ahead (12 = 1 hour)
        test_windows: Number of prediction windows to test
        
    Returns:
        dict with accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {symbol}")
    print(f"{'='*60}")
    
    # Fetch historical data
    print(f"Fetching data for {symbol}...")
    df = fetch_latest_market_data(symbol, period='7d', interval='5m')
    
    if df is None:
        print(f"❌ No data for {symbol}")
        return None
    
    # Adjust lookback if we don't have enough data
    min_required = lookback + pred_len + (test_windows * pred_len)
    if len(df) < min_required:
        # Reduce lookback to fit available data
        lookback = max(200, len(df) - pred_len - (test_windows * pred_len))
        print(f"⚠️ Limited data ({len(df)} candles), reducing lookback to {lookback}")
        
        if len(df) < 200 + pred_len:
            print(f"❌ Insufficient data for {symbol} (need at least 200+{pred_len})")
            return None
    
    print(f"✅ Got {len(df)} candles")
    
    # Walk forward testing
    correct = 0
    total = 0
    predictions = []
    
    # Test every hour (12 candles) to avoid overlap
    step = pred_len
    start_idx = lookback
    end_idx = len(df) - pred_len
    
    for i in range(start_idx, end_idx, step):
        if total >= test_windows:
            break
        
        try:
            # Historical data up to this point
            historical = df.iloc[:i].copy()
            
            # Actual future data
            future = df.iloc[i:i+pred_len].copy()
            
            # Get Kronos prediction
            signal = predictor.get_directional_signal(historical, lookback=lookback, pred_len=pred_len)
            
            if 'error' in signal:
                print(f"  Window {total+1}: Prediction failed - {signal['error']}")
                continue
            
            # Calculate actual price movement
            actual_start = float(historical['close'].iloc[-1])
            actual_end = float(future['close'].iloc[-1])
            actual_change = (actual_end - actual_start) / actual_start
            
            # Check if prediction was correct
            predicted_signal = signal['signal']
            predicted_change = signal['predicted_change']
            
            if predicted_signal == 'LONG' and actual_change > 0:
                correct += 1
                result = '✅'
            elif predicted_signal == 'SHORT' and actual_change < 0:
                correct += 1
                result = '✅'
            elif predicted_signal == 'NEUTRAL':
                # Don't count neutral
                continue
            else:
                result = '❌'
            
            total += 1
            
            predictions.append({
                'window': total,
                'predicted_signal': predicted_signal,
                'predicted_change': predicted_change,
                'actual_change': actual_change,
                'correct': result == '✅',
                'confidence': signal['confidence']
            })
            
            # Print progress
            accuracy_so_far = (correct / total * 100) if total > 0 else 0
            print(f"  Window {total:2d}: {result} Pred: {predicted_signal:7s} ({predicted_change:+.2%}), Actual: {actual_change:+.2%}, Acc: {accuracy_so_far:.1f}%")
            
        except Exception as e:
            print(f"  Window {total+1}: Error - {e}")
            continue
    
    # Calculate metrics
    if total == 0:
        print(f"❌ No valid predictions for {symbol}")
        return None
    
    accuracy = (correct / total) * 100
    
    # Separate by signal type
    long_preds = [p for p in predictions if p['predicted_signal'] == 'LONG']
    short_preds = [p for p in predictions if p['predicted_signal'] == 'SHORT']
    
    long_accuracy = (sum(1 for p in long_preds if p['correct']) / len(long_preds) * 100) if long_preds else 0
    short_accuracy = (sum(1 for p in short_preds if p['correct']) / len(short_preds) * 100) if short_preds else 0
    
    # Average confidence
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    
    results = {
        'symbol': symbol,
        'total_predictions': total,
        'correct': correct,
        'accuracy': accuracy,
        'long_count': len(long_preds),
        'long_accuracy': long_accuracy,
        'short_count': len(short_preds),
        'short_accuracy': short_accuracy,
        'avg_confidence': avg_confidence,
        'predictions': predictions
    }
    
    # Print summary
    print(f"\n--- Results for {symbol} ---")
    print(f"Total predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"LONG signals: {len(long_preds)} ({long_accuracy:.1f}% accuracy)")
    print(f"SHORT signals: {len(short_preds)} ({short_accuracy:.1f}% accuracy)")
    print(f"Avg confidence: {avg_confidence:.1%}")
    
    return results

def main():
    """Main evaluation function."""
    print("="*60)
    print("KRONOS DIRECTIONAL ACCURACY EVALUATION")
    print("="*60)
    
    # Check if Kronos is available
    if not is_kronos_available():
        print("❌ Kronos not available - missing dependencies")
        return
    
    # Initialize predictor
    print("\nInitializing Kronos predictor (this may take a moment)...")
    predictor = BigBotKronosPredictor(model_size='small', device='cpu')
    print("✅ Predictor ready")
    
    # Symbols to test
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Test parameters (adjusted for data availability)
    lookback = 300  # Historical candles (reduced from 400 due to data limits)
    pred_len = 12   # Predict 12 candles ahead (1 hour)
    test_windows = 10  # Test 10 prediction windows per symbol (reduced from 20)
    
    print(f"\nTest parameters:")
    print(f"  Lookback: {lookback} candles")
    print(f"  Prediction horizon: {pred_len} candles (1 hour)")
    print(f"  Test windows: {test_windows} per symbol")
    print(f"  Symbols: {', '.join(symbols)}")
    
    # Run evaluation
    all_results = []
    
    for symbol in symbols:
        result = evaluate_symbol(predictor, symbol, lookback, pred_len, test_windows)
        if result:
            all_results.append(result)
    
    # Overall summary
    if not all_results:
        print("\n❌ No valid results")
        return
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    
    total_predictions = sum(r['total_predictions'] for r in all_results)
    total_correct = sum(r['correct'] for r in all_results)
    overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"\nAcross all symbols:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Correct: {total_correct}")
    print(f"  Overall accuracy: {overall_accuracy:.1f}%")
    
    print(f"\nPer symbol:")
    for r in all_results:
        print(f"  {r['symbol']:10s}: {r['accuracy']:.1f}% ({r['correct']}/{r['total_predictions']})")
    
    # Decision
    print(f"\n{'='*60}")
    print("DECISION")
    print(f"{'='*60}")
    
    if overall_accuracy >= 60:
        print(f"✅ INTEGRATE - {overall_accuracy:.1f}% accuracy exceeds 60% threshold")
        print("   Recommendation: Add Kronos to bot_v2_safe.py")
    elif overall_accuracy >= 50:
        print(f"⚠️ MARGINAL - {overall_accuracy:.1f}% accuracy is 50-60%")
        print("   Recommendation: Try Kronos-base model or fine-tuning")
    else:
        print(f"❌ REJECT - {overall_accuracy:.1f}% accuracy below 50%")
        print("   Recommendation: Don't use Kronos for 5-min crypto")
    
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
