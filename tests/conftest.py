"""Configuration for pytest."""

import sys
import os

# Add src to path for imports in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
