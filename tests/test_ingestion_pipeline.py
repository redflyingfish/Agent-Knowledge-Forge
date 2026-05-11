import asyncio

from agent_knowledge_harvester.config import settings
from agent_knowledge_harvester.ingestion.pipeline import IngestionPipeline


class SlowReader:
    async def fetch(self, target):  # noqa: ANN001
        await asyncio.sleep(0.2)
        raise AssertionError("timeout should cancel the slow fetch")


def test_ingestion_pipeline_times_out_single_url(tmp_path) -> None:
    runtime_settings = settings.model_copy(update={"ingestion_timeout_seconds": 0.01})
    pipeline = IngestionPipeline(runtime_settings, concurrency=1)
    pipeline.jina_reader = SlowReader()

    results = asyncio.run(
        pipeline.ingest_urls(["https://example.com/slow"], out_dir=tmp_path)
    )

    assert len(results) == 1
    assert not results[0].success
    assert "timed out" in (results[0].error or "")
    assert (tmp_path / "run_stats.json").exists()

