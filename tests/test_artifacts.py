from agent_knowledge_harvester.utils.artifacts import find_scratch_artifacts, remove_paths


def test_find_scratch_artifacts_only_returns_disposable_dirs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "analysis-empty-check").mkdir(parents=True)
    (data_dir / "smoke").mkdir(parents=True)
    (data_dir / "smoke-trending").mkdir()
    (data_dir / "analysis-smoke").mkdir()

    artifacts = find_scratch_artifacts(data_dir)

    assert [path.name for path in artifacts] == [
        "analysis-empty-check",
        "smoke",
        "smoke-trending",
    ]


def test_remove_paths_deletes_nested_artifacts(tmp_path) -> None:
    artifact = tmp_path / "smoke"
    nested = artifact / "nested"
    nested.mkdir(parents=True)
    (nested / "raw.json").write_text("{}", encoding="utf-8")

    removed = remove_paths([artifact])

    assert removed == 1
    assert not artifact.exists()
