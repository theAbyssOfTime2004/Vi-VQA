"""Streaming dataset loading (used to keep Colab/Kaggle disk usage down)."""

import sys
import types

import pytest

from vivqa.config import Config
from vivqa.data.prepare import load_records, prepare


@pytest.fixture
def fake_datasets(monkeypatch):
    """Stand in for the `datasets` package and record how it was called."""
    calls = []

    def load_dataset(name, split=None, streaming=False, **kwargs):
        calls.append({"name": name, "split": split, "streaming": streaming})
        return [{"id": i, "description": "Mô tả.", "conversations": [
            {"role": "user", "content": f"Câu hỏi {i}?"},
            {"role": "assistant", "content": f"Trả lời {i}."},
        ]} for i in range(10)]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


class TestLoadRecords:
    def test_streaming_off_by_default(self, fake_datasets):
        load_records(Config().data)
        assert fake_datasets[0]["streaming"] is False

    def test_streaming_flag_is_forwarded(self, fake_datasets):
        data = Config().data
        data.streaming = True
        load_records(data, limit=50)
        assert fake_datasets[0]["streaming"] is True
        assert fake_datasets[0]["split"] == "train"

    def test_streaming_without_limit_warns(self, fake_datasets, caplog):
        # Streaming with no limit still walks everything; it only saves the
        # up-front download, and the run will not look any faster.
        data = Config().data
        data.streaming = True
        load_records(data, limit=None)
        assert "still iterates the whole split" in caplog.text

    def test_streaming_with_limit_does_not_warn(self, fake_datasets, caplog):
        data = Config().data
        data.streaming = True
        load_records(data, limit=10)
        assert "still iterates" not in caplog.text


class TestPrepareUsesLoadRecords:
    def test_prepare_streams_when_configured(self, fake_datasets, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.data.streaming = True

        counts = prepare(config, limit=5)

        assert fake_datasets[0]["streaming"] is True
        assert counts["records"] == 5

    def test_injected_records_skip_the_loader_entirely(self, fake_datasets, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")

        prepare(config, records=[{"id": 0, "description": "x", "conversations": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}])

        assert fake_datasets == []


class TestStreamingConfig:
    def test_config_accepts_the_flag(self, repo_config):
        from vivqa.config import load_config

        config = load_config(repo_config, overrides=["data.streaming=true"])
        assert config.data.streaming is True

    def test_repo_config_defaults_to_off(self, repo_config):
        from vivqa.config import load_config

        assert load_config(repo_config).data.streaming is False
