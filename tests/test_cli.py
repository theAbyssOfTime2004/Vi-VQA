"""CLI argument handling and failure modes."""

import json

import pytest

from fvqa.cli import main


class TestConfigCommand:
    def test_prints_resolved_config_as_json(self, capsys, repo_config):
        assert main(["config", "--config", repo_config]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["model"]["model_id"].startswith("Qwen/Qwen3-VL")

    def test_overrides_are_visible_in_the_output(self, capsys, repo_config):
        main(["config", "--config", repo_config, "--set", "training.num_train_epochs=9"])
        printed = json.loads(capsys.readouterr().out)
        assert printed["training"]["num_train_epochs"] == 9


class TestFailureModes:
    def test_invalid_config_exits_with_two(self, repo_config):
        # Exit 2 distinguishes "you configured it wrong" from "the run failed".
        assert main(["config", "--config", repo_config, "--set", "data.splits.val=0.9"]) == 2

    def test_missing_config_file_exits_with_two(self, tmp_path):
        assert main(["config", "--config", str(tmp_path / "absent.yaml")]) == 2

    def test_malformed_override_exits_with_two(self, repo_config):
        assert main(["config", "--config", repo_config, "--set", "nonsense"]) == 2

    def test_train_without_data_exits_with_one(self, repo_config, tmp_path):
        code = main(
            ["train", "--config", repo_config, "--dry-run", "--set", f"data.data_dir={tmp_path}"]
        )
        assert code == 1

    def test_unknown_command_is_rejected(self, repo_config):
        with pytest.raises(SystemExit):
            main(["teleport", "--config", repo_config])
