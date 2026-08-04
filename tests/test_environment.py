"""Smoke tests for the local development environment."""

import sys


def test_supported_python_version() -> None:
    """The project requires Python 3.11 or 3.12."""
    assert (3, 11) <= sys.version_info[:2] < (3, 13)


def test_core_dependencies_import() -> None:
    """Core application dependencies should be importable."""
    import fastapi
    import httpx
    import pydantic_settings
    import uvicorn

    assert fastapi.__version__
    assert httpx.__version__
    assert pydantic_settings.__version__
    assert uvicorn.__version__
