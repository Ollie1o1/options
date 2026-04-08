#!/usr/bin/env python3
"""
Options Screener Entry Point (Hardened)
Delegates to modular CLI and scanner.
"""

import sys
import os

# Ensure src is in path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .cli import main

if __name__ == "__main__":
    main()
