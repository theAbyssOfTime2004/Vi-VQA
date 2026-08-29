"""`ensure_trainer_repo`'s clone/pin/checkout logic.

These tests use a throwaway local git repository as the "remote" instead
of hitting GitHub — the point under test is the git plumbing (shallow
fetch of a pinned commit, fallback to a full clone, warning on drift),
not network access.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

from fvqa.train.runner import ensure_trainer_repo, trainer_env


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture(scope="session")
def fake_trainer_remote(tmp_path_factory):
    """A local repo with two commits, the entry point added in the second.

    Session-scoped: every test only ever clones *from* this repo, never
    writes to it, and building it costs several seconds of `git commit`.
    """
    remote = tmp_path_factory.mktemp("remote")
    _git("init", "-q", cwd=remote)
    _git("config", "user.email", "test@example.com", cwd=remote)
    _git("config", "user.name", "test", cwd=remote)

    (remote / "README.md").write_text("v1\n")
    _git("add", ".", cwd=remote)
    _git("commit", "-q", "-m", "v1, no entry point yet", cwd=remote)
    old_revision = _git("rev-parse", "HEAD", cwd=remote).stdout.strip()

    train_dir = remote / "src" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "train_sft.py").write_text("# entry point\n")
    _git("add", ".", cwd=remote)
    _git("commit", "-q", "-m", "v2, adds entry point", cwd=remote)
    new_revision = _git("rev-parse", "HEAD", cwd=remote).stdout.strip()

    return {"url": str(remote), "old": old_revision, "new": new_revision}


class TestFreshClone:
    def test_pins_to_the_given_revision(self, tmp_path, fake_trainer_remote):
        dest = tmp_path / "trainer"
        ensure_trainer_repo(
            str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )
        head = _git("rev-parse", "HEAD", cwd=dest).stdout.strip()
        assert head == fake_trainer_remote["new"]

    def test_pinning_to_an_older_revision_omits_the_entry_point(
        self, tmp_path, fake_trainer_remote
    ):
        # The entry point only exists as of the newer commit — pinning to
        # the older one should faithfully check that commit out, not the
        # branch tip, and therefore raise because the entry point is missing.
        dest = tmp_path / "trainer"
        with pytest.raises(RuntimeError, match="does not contain"):
            ensure_trainer_repo(
                str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["old"]
            )

    def test_without_a_pinned_revision_clones_the_branch_tip(self, tmp_path, fake_trainer_remote):
        dest = tmp_path / "trainer"
        ensure_trainer_repo(str(dest), url=fake_trainer_remote["url"], revision=None)
        head = _git("rev-parse", "HEAD", cwd=dest).stdout.strip()
        assert head == fake_trainer_remote["new"]

    def test_returns_the_absolute_destination_path(self, tmp_path, fake_trainer_remote):
        dest = tmp_path / "trainer"
        result = ensure_trainer_repo(
            str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )
        assert result == str(dest.resolve())


class TestExistingCheckout:
    def test_already_on_the_pinned_revision_is_a_no_op(self, tmp_path, fake_trainer_remote):
        dest = tmp_path / "trainer"
        ensure_trainer_repo(
            str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )
        # A second call with the same pin must not fail or need network
        # access again (the "remote" still exists here, but nothing new
        # should be fetched from it for this to succeed).
        result = ensure_trainer_repo(
            str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )
        assert result == str(dest.resolve())

    def test_warns_but_does_not_raise_when_on_a_different_revision(
        self, tmp_path, fake_trainer_remote, caplog
    ):
        dest = tmp_path / "trainer"
        ensure_trainer_repo(
            str(dest), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )

        with caplog.at_level(logging.WARNING):
            ensure_trainer_repo(
                str(dest),
                url=fake_trainer_remote["url"],
                # A revision that does not exist locally and was never
                # fetched — simulates upstream having moved on.
                revision="0" * 40,
            )
        assert "not the pinned revision" in caplog.text

    def test_missing_entry_point_raises_even_when_already_cloned(self, tmp_path):
        dest = tmp_path / "trainer"
        dest.mkdir()
        _git("init", "-q", cwd=dest)
        with pytest.raises(RuntimeError, match="does not contain"):
            ensure_trainer_repo(str(dest), url="unused", revision=None)


class TestTrainerEnv:
    """The trainer's entry point is not importable without its own src/.

    `train_sft.py` imports `model`, `trainer`, `dataset` and `params` as
    top-level packages. Every one of the trainer's `scripts/*.sh` starts
    with `export PYTHONPATH=src:$PYTHONPATH`; launching it without that
    dies on the first import line with "No module named 'model'", after
    deepspeed has already spun up.
    """

    def test_puts_the_trainers_src_on_pythonpath(self):
        env = trainer_env("/opt/trainer", base={})
        assert env["PYTHONPATH"] == os.path.join("/opt/trainer", "src")

    def test_prepends_rather_than_replacing(self):
        # On Modal PYTHONPATH already carries the fvqa package; replacing
        # it would swap this failure for a different one.
        env = trainer_env("/opt/trainer", base={"PYTHONPATH": "/root/src"})
        assert env["PYTHONPATH"] == f"/opt/trainer/src{os.pathsep}/root/src"

    def test_keeps_the_rest_of_the_environment(self):
        env = trainer_env("/opt/trainer", base={"HF_TOKEN": "x", "CUDA_HOME": "/usr/local/cuda"})
        assert env["HF_TOKEN"] == "x"
        assert env["CUDA_HOME"] == "/usr/local/cuda"

    def test_defaults_to_the_real_environment(self, monkeypatch):
        monkeypatch.setenv("FVQA_TEST_MARKER", "present")
        assert trainer_env("/opt/trainer")["FVQA_TEST_MARKER"] == "present"


class TestRunTrainingLaunch:
    """What `run_training` actually hands to the subprocess."""

    def _config(self, tmp_path, trainer_dir):
        from fvqa.config import Config

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.json").write_text("[]", encoding="utf-8")
        images = tmp_path / "images"
        images.mkdir()

        config = Config()
        config.data.data_dir = str(data_dir)
        config.data.image_folder = str(images)
        config.training.output_dir = str(tmp_path / "out")
        config.trainer.revision = ""
        return config

    def test_launches_with_the_trainers_src_on_pythonpath(
        self, tmp_path, fake_trainer_remote, monkeypatch
    ):
        import fvqa.train.runner as runner

        trainer_dir = tmp_path / "trainer"
        ensure_trainer_repo(
            str(trainer_dir), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )

        captured = {}

        def fake_run(command, cwd=None, env=None):
            captured["command"] = command
            captured["cwd"] = cwd
            captured["env"] = env

            class Result:
                returncode = 0

            return Result()

        monkeypatch.setattr(runner.subprocess, "run", fake_run)

        config = self._config(tmp_path, trainer_dir)
        exit_code = runner.run_training(config, trainer_dir=str(trainer_dir))

        assert exit_code == 0
        assert captured["cwd"] == str(trainer_dir.resolve())
        expected = os.path.join(str(trainer_dir.resolve()), "src")
        assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == expected

    def test_dry_run_launches_nothing(self, tmp_path, fake_trainer_remote, monkeypatch):
        import fvqa.train.runner as runner

        trainer_dir = tmp_path / "trainer"
        ensure_trainer_repo(
            str(trainer_dir), url=fake_trainer_remote["url"], revision=fake_trainer_remote["new"]
        )

        def fail(*args, **kwargs):
            raise AssertionError("dry_run must not launch the trainer")

        monkeypatch.setattr(runner.subprocess, "run", fail)

        config = self._config(tmp_path, trainer_dir)
        assert runner.run_training(config, trainer_dir=str(trainer_dir), dry_run=True) == 0
