#!/usr/bin/env python3
"""Entry point for the Clarify LSP server."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clarify_lsp.server import main

if __name__ == "__main__":
    main()
