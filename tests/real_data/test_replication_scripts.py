import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.real_data]


def test_replication_material_scripts_are_present_and_syntax_valid():
    scripts_dir = Path("replication_material/scripts")
    scripts = ["basic_usage.py", "benchmark.py", "real_scenario.py"]
    missing_scripts = [script for script in scripts if not (scripts_dir / script).exists()]
    if missing_scripts:
        pytest.skip(f"replication scripts not available: {missing_scripts}")

    for script in scripts:
        path = scripts_dir / script
        ast.parse(path.read_text(), filename=str(path))


def test_replication_material_includes_local_heart_disease_data():
    data_dir = Path("replication_material/data/heart_disease")
    expected = [data_dir / f"processed.{site}.data" for site in ["cleveland", "hungarian", "switzerland", "va"]]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        pytest.skip(f"replication heart disease data not available: {missing}")

    assert all(path.stat().st_size > 0 for path in expected)
