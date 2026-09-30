from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import api.routes.documents as documents_route


class FakeKnowledgeBase:

    def __init__(self):
        self.added_files = []

    def add_file(
        self,
        file_path,
        original_filename=None,
    ):
        result = {
            "document_id": (
                f"id-{len(self.added_files) + 1}"
            ),
            "filename": original_filename,
            "stored_path": str(file_path),
            "text_chunks": (
                0
                if file_path.suffix == ".png"
                else 1
            ),
            "image_chunks": (
                1
                if file_path.suffix == ".png"
                else 0
            ),
        }

        self.added_files.append(result)

        return result

    @property
    def document_count(self):
        return len(self.added_files)

    @property
    def text_chunk_count(self):
        return sum(
            item["text_chunks"]
            for item in self.added_files
        )

    @property
    def image_chunk_count(self):
        return sum(
            item["image_chunks"]
            for item in self.added_files
        )


class FakeResearchAgent:

    def __init__(self):
        self.rag = object()
        self.knowledge_base = (
            FakeKnowledgeBase()
        )


class FakeRegistry:

    def __init__(self, research_agent):
        self.research_agent = (
            research_agent
        )

    def get(self, task_type):

        if task_type == "retrieve":
            return self.research_agent

        return None


class FakeSupervisor:

    def __init__(self, registry):
        self.registry = registry


class FakeAgent:

    def __init__(self, supervisor):
        self.supervisor = supervisor


def create_test_image() -> bytes:

    image = Image.new(
        mode="RGB",
        size=(100, 100),
        color="blue",
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def build_test_client(
    research_agent,
):

    agent = FakeAgent(
        supervisor=FakeSupervisor(
            registry=FakeRegistry(
                research_agent
            )
        )
    )

    app = FastAPI()

    app.include_router(
        documents_route.router
    )

    app.state.rag = agent

    return TestClient(app)


def test_sequential_uploads_are_accumulated(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_route,
        "UPLOAD_DIR",
        tmp_path,
    )

    research_agent = FakeResearchAgent()

    original_pipeline = research_agent.rag

    client = build_test_client(
        research_agent
    )

    first_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "first.txt",
                b"First document.",
                "text/plain",
            )
        },
    )

    second_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "second.txt",
                b"Second document.",
                "text/plain",
            )
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert (
        research_agent
        .knowledge_base
        .document_count
        == 2
    )

    assert (
        research_agent
        .knowledge_base
        .text_chunk_count
        == 2
    )

    # Pipeline was not replaced.
    assert research_agent.rag is original_pipeline


def test_multiple_files_in_one_request(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_route,
        "UPLOAD_DIR",
        tmp_path,
    )

    research_agent = FakeResearchAgent()

    client = build_test_client(
        research_agent
    )

    response = client.post(
        "/documents/upload",
        files=[
            (
                "files",
                (
                    "profile.txt",
                    b"Ali is an AI engineer.",
                    "text/plain",
                ),
            ),
            (
                "files",
                (
                    "blue_square.png",
                    create_test_image(),
                    "image/png",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["uploaded_count"] == 2

    assert len(data["files"]) == 2

    assert (
        data["knowledge_base"][
            "document_count"
        ]
        == 2
    )

    assert (
        data["knowledge_base"][
            "text_chunk_count"
        ]
        == 1
    )

    assert (
        data["knowledge_base"][
            "image_chunk_count"
        ]
        == 1
    )

    filenames = {
        item["filename"]
        for item in data["files"]
    }

    assert filenames == {
        "profile.txt",
        "blue_square.png",
    }


def test_invalid_batch_is_rejected_before_ingestion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_route,
        "UPLOAD_DIR",
        tmp_path,
    )

    research_agent = FakeResearchAgent()

    client = build_test_client(
        research_agent
    )

    response = client.post(
        "/documents/upload",
        files=[
            (
                "files",
                (
                    "valid.txt",
                    b"Valid document.",
                    "text/plain",
                ),
            ),
            (
                "files",
                (
                    "invalid.zip",
                    b"Invalid archive.",
                    "application/zip",
                ),
            ),
        ],
    )

    assert response.status_code == 400

    assert (
        research_agent
        .knowledge_base
        .document_count
        == 0
    )

    assert list(tmp_path.iterdir()) == []