#!/usr/bin/env python3
"""Entry-point wrapper for the options screener.

This thin script delegates to the implementation in ``src.options_screener``
so existing workflows that call ``python options_screener.py`` continue to
work after the project was reorganized into a src/ layout.
"""

from src.options_screener import main


if __name__ == "__main__":
    main()
