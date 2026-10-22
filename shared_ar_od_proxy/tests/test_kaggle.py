import importlib
import json
from pathlib import Path

import pytest


def test_notebook_pins_sources_and_compiles(tmp_path):
    kaggle = importlib.import_module("ar_od_proxy.kaggle")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "payload"
    kaggle.prepare(root, directory, "a" * 40, "example", "shared-ar-od-pilot")
    metadata, job = kaggle.verify_payload(directory, "example/shared-ar-od-pilot")
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert job["commit"] == "a" * 40
    notebook = json.loads((directory / "notebook.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
    assert job["source_hashes"]["ar_od_proxy/models.py"]


def test_payload_tampering_is_rejected(tmp_path):
    kaggle = importlib.import_module("ar_od_proxy.kaggle")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "payload"
    kaggle.prepare(root, directory, "a" * 40, "example", "shared-ar-od-pilot")
    (directory / "notebook.ipynb").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        kaggle.verify_payload(directory, "example/shared-ar-od-pilot")


def test_mutable_branch_cannot_be_used_as_notebook_pin(tmp_path):
    kaggle = importlib.import_module("ar_od_proxy.kaggle")
    with pytest.raises(ValueError, match="commit"):
        kaggle.prepare(Path(__file__).resolve().parents[1], tmp_path / "payload", "main", "example", "shared-ar-od-pilot")


def test_extra_qp_is_rejected_by_protocol(tmp_path):
    runner = importlib.import_module("ar_od_proxy.run")
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"qps": [30, 35, 40, 45, 50], "codecs": ["h264", "h265"]}))
    with pytest.raises(ValueError, match="exactly"):
        runner.configuration(config)


def test_installed_default_config_exists():
    runner = importlib.import_module("ar_od_proxy.run")
    assert runner.configuration(None)["qps"] == [30, 35, 40, 45]


def test_zero_exit_cli_error_is_not_a_submission_acknowledgment():
    kaggle = importlib.import_module("ar_od_proxy.kaggle")
    for output in ("Kernel push error: GPU quota exceeded", "Invalid dataset source: missing/dataset", "",
                   "Kernel version 0 successfully pushed. Please check progress at https://www.kaggle.com/code/example/shared-ar-od-pilot"):
        with pytest.raises(ValueError, match="submission"):
            kaggle.validate_submission(output, "example/shared-ar-od-pilot")
    receipt = kaggle.validate_submission(
        "Kernel version 3 successfully pushed. Please check progress at https://www.kaggle.com/code/example/shared-ar-od-pilot",
        "example/shared-ar-od-pilot")
    assert receipt["version"] == 3 and receipt["running_verified"] is False
def test_calibration_notebook_mounts_verified_baseline_and_runs_calibration(tmp_path):
    from pathlib import Path
    import json
    from ar_od_proxy import kaggle
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "calibration"
    baseline = "b" * 64
    kaggle.prepare(root, directory, "a" * 40, "example", "calibration-pilot",
                   action="calibrate", baseline_sha256=baseline)
    meta = json.loads((directory / "kernel-metadata.json").read_text())
    notebook = json.loads((directory / "notebook.ipynb").read_text())
    code = "".join(notebook["cells"][0]["source"])
    assert meta["kernel_sources"] == ["qktttttttttt/ar-od-proxy-qp4-20261022"]
    assert "ar_od_proxy.calibration" in code and baseline in code
    assert "baseline checkpoint hash mismatch" in code

