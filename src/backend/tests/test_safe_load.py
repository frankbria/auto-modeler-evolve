"""storage.safe_load confines joblib.load to the data root (#21).

Model/pipeline artifacts are pickles — joblib.load runs arbitrary code on the
payload. safe_load rejects any path that resolves outside DATA_DIR before
deserializing, as defense-in-depth against a path-confusion -> pickle RCE.
"""

import joblib
import pytest

from core import storage
from core.path_safety import UnsafePathError


def test_safe_load_reads_artifact_inside_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    models = storage.models_dir()
    models.mkdir(parents=True)
    target = models / "m.joblib"
    joblib.dump({"ok": True}, target)

    assert storage.safe_load(target) == {"ok": True}


def test_safe_load_rejects_path_outside_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    outside = tmp_path / "evil.joblib"
    joblib.dump({"payload": "x"}, outside)

    with pytest.raises(UnsafePathError):
        storage.safe_load(outside)


def test_safe_load_rejects_traversal_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    storage.models_dir().mkdir(parents=True)
    escape = storage.models_dir() / ".." / ".." / "etc_passwd.joblib"

    with pytest.raises(UnsafePathError):
        storage.safe_load(escape)
