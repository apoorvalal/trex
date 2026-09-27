#!/usr/bin/env python3
"""Compatibility entry point for the Quarto API source generator."""
from build import main

if __name__ == "__main__":
    main(["--api-only"])
