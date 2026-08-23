import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def repo_config():
    """Path to the repository's own config.yaml."""
    return os.path.join(REPO_ROOT, "config", "config.yaml")
