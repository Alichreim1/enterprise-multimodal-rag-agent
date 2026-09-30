from pathlib import Path
import logging
import shutil
import uuid

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


UPLOAD_DIR = Path("data/uploads")


TEXT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


ALLOWED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
)


# =========================================================
# Get Research Agent
# =========================================================

def get_research_agent(agent):

    research_agent = (
        agent
        .supervisor
        .registry
        .get("retrieve")
    )

    if research_agent is None:
        raise RuntimeError(
            "ResearchAgent is not registered "
            "for task type 'retrieve'."
        )

    if (
        not hasattr(
            research_agent,
            "knowledge_base",
        )
        or research_agent.knowledge_base is None
    ):
        raise RuntimeError(
            "ResearchAgent does not have an "
            "active KnowledgeBaseManager."
        )

    return research_agent


# =========================================================
# Validate Upload
# =========================================================

def validate_upload(
    upload: UploadFile,
) -> str:

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required.",
        )

    extension = Path(
        upload.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                "Allowed types: PDF, DOCX, TXT, "
                "PNG, JPG, JPEG, WEBP."
            ),
        )

    return extension


# =========================================================
# Save Uploaded File
# =========================================================

def save_upload(
    upload: UploadFile,
    extension: str,
) -> Path:

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = (
        UPLOAD_DIR / stored_filename
    )

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            upload.file,
            buffer,
        )

    return file_path


# =========================================================
# Upload Documents And Images
# =========================================================

@router.post("/upload")
def upload_documents(
    request: Request,

    # New multi-file field
    files: list[UploadFile] | None = File(
        default=None
    ),

    # Old single-file field, kept for compatibility
    file: UploadFile | None = File(
        default=None
    ),
):
    try:
        # ==================================================
        # Collect Single And Multiple Inputs
        # ==================================================

        uploads: list[UploadFile] = []

        if file is not None:
            uploads.append(file)

        if files:
            uploads.extend(files)

        if not uploads:
            raise HTTPException(
                status_code=400,
                detail=(
                    "At least one file is required."
                ),
            )

        # ==================================================
        # Validate Entire Batch Before Saving Anything
        # ==================================================

        validated_uploads = [
            (
                upload,
                validate_upload(upload),
            )
            for upload in uploads
        ]

        # ==================================================
        # Active Knowledge Base
        # ==================================================

        agent = request.app.state.rag

        research_agent = get_research_agent(
            agent
        )

        knowledge_base = (
            research_agent.knowledge_base
        )

        # ==================================================
        # Save And Add Files
        # ==================================================

        uploaded_results = []

        for upload, extension in validated_uploads:

            file_path = save_upload(
                upload=upload,
                extension=extension,
            )

            ingestion_result = (
                knowledge_base.add_file(
                    file_path=file_path,
                    original_filename=(
                        upload.filename
                    ),
                )
            )

            content_type = (
                "image"
                if extension
                in IMAGE_EXTENSIONS
                else "document"
            )

            uploaded_results.append(
                {
                    "filename": upload.filename,
                    "stored_filename": (
                        file_path.name
                    ),
                    "content_type": (
                        content_type
                    ),
                    "document_id": (
                        ingestion_result[
                            "document_id"
                        ]
                    ),
                    "text_chunks": (
                        ingestion_result[
                            "text_chunks"
                        ]
                    ),
                    "image_chunks": (
                        ingestion_result[
                            "image_chunks"
                        ]
                    ),
                }
            )

            logger.info(
                "File added to knowledge base: %s",
                upload.filename,
            )

        # ==================================================
        # Response
        # ==================================================

        response = {
            "success": True,
            "uploaded_count": len(
                uploaded_results
            ),
            "files": uploaded_results,
            "knowledge_base": {
                "document_count": (
                    knowledge_base.document_count
                ),
                "text_chunk_count": (
                    knowledge_base.text_chunk_count
                ),
                "image_chunk_count": (
                    knowledge_base.image_chunk_count
                ),
            },
            "message": (
                f"{len(uploaded_results)} "
                "file(s) uploaded successfully."
            ),
        }

        # ==================================================
        # Backward-Compatible Single File Fields
        # ==================================================

        if len(uploaded_results) == 1:

            single_result = (
                uploaded_results[0]
            )

            response.update(
                {
                    "filename": (
                        single_result[
                            "filename"
                        ]
                    ),
                    "stored_filename": (
                        single_result[
                            "stored_filename"
                        ]
                    ),
                    "content_type": (
                        single_result[
                            "content_type"
                        ]
                    ),
                }
            )

        return response

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "File upload failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process uploaded "
                "file(s)."
            ),
        ) from exc