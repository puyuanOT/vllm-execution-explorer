from pathlib import Path

from runner.observability import ExperimentTrace, source_identity


def test_unconfigured_trace_has_no_network_client(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    trace = ExperimentTrace(tmp_path, "Qwen", {"profiling": "off"})
    trace.start_request("a", "A prompt", [1, 2])
    trace.finish_request("a", "A response", [3])
    trace.finish("completed")
    assert trace.metadata()["status"] == "unconfigured"
    assert trace.metadata()["trace_id"] is None


def test_explicit_trace_handles_preserve_hierarchy_content_and_incomplete_calls(monkeypatch, tmp_path):
    import langsmith.run_trees

    created = []

    class FakeRunTree:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.children = []
            self.id = "test-trace"
            self.client = self
            self.flushed = None
            self.ending = None
            created.append(self)

        def post(self):
            pass

        def create_child(self, **kwargs):
            child = FakeRunTree(**kwargs)
            self.children.append(child)
            return child

        def end(self, **kwargs):
            self.ending = kwargs

        def patch(self):
            pass

        def flush(self, timeout):
            self.flushed = timeout

    monkeypatch.setattr(langsmith.run_trees, "RunTree", FakeRunTree)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-never-transmitted")
    trace = ExperimentTrace(tmp_path, "Qwen", {"model_revision": "immutable"})
    trace.start_request("a", "Actual prompt", [1, 2], request_configuration={"sampling": {"temperature": 0.2}, "priority": 1})
    trace.start_request("b", "Incomplete prompt", [7], request_configuration={"sampling": {"temperature": 0.9}, "priority": -2})
    trace.finish_request("a", "Actual response", [3, 4])
    trace.finish("cancelled")
    root, capture, complete, incomplete = created
    assert root.children == [capture]
    assert capture.children == [complete, incomplete]
    assert complete.kwargs["run_type"] == "llm"
    assert complete.kwargs["inputs"]["messages"] == [{"role": "user", "content": "Actual prompt"}]
    assert complete.kwargs["inputs"]["token_ids"] == [1, 2]
    assert complete.kwargs["inputs"]["request_configuration"]["sampling"]["temperature"] == 0.2
    assert incomplete.kwargs["inputs"]["request_configuration"]["priority"] == -2
    assert root.kwargs["extra"]["metadata"]["code_identity"]["source_file_count"] > 0
    assert complete.ending["outputs"] == {"generations": [{"text": "Actual response"}], "token_ids": [3, 4]}
    assert "did not complete" in incomplete.ending["error"]
    assert root.flushed == 5.0
    assert trace.metadata()["status"] == "queued"
    assert "test-only-never-transmitted" not in str([run.kwargs for run in created])


def test_sdk_failure_detail_does_not_leak_credentials(monkeypatch, tmp_path):
    import langsmith.run_trees

    def failure(**kwargs):
        raise RuntimeError("Authorization: private-test-value")

    monkeypatch.setattr(langsmith.run_trees, "RunTree", failure)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-never-transmitted")
    trace = ExperimentTrace(Path(tmp_path), "Qwen", {})
    assert trace.metadata()["status"] == "failed"
    assert "private-test-value" not in str(trace.metadata())


def test_source_identity_changes_with_capture_source_but_not_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_EXPLORER_CODE_REVISION", raising=False)
    (tmp_path / "runner").mkdir()
    (tmp_path / "instrumentation").mkdir()
    source = tmp_path / "runner" / "capture.py"
    source.write_text("version = 1\n")
    first = source_identity(tmp_path)
    assert first["source_file_count"] == 1
    assert first["declared_revision"] is None
    assert len(first["source_sha256"]) == 64
    (tmp_path / "notes.txt").write_text("not distributed capture code")
    assert source_identity(tmp_path) == first
    source.write_text("version = 2\n")
    assert source_identity(tmp_path)["source_sha256"] != first["source_sha256"]
