import subprocess
import sys

import pytest


@pytest.mark.parametrize("directory", ["app", "scripts"])
def test_can_import_settings_from(directory):
    result = subprocess.run(
        [sys.executable, "-c", "from backend.config import Settings"],
        cwd=directory,
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_integration_marker_is_registered(pytestconfig):
    assert any(
        line.startswith("integration:") for line in pytestconfig.getini("markers")
    )
