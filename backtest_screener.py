#!/usr/bin/env python3
"""Entry-point wrapper for the backtesting engine.

This script delegates to ``src.backtest_screener.main`` so existing calls to
``python backtest_screener.py`` remain valid after adopting a src/ layout.
"""

from src.backtest_screener import main


if __name__ == "__main__":
    main()
