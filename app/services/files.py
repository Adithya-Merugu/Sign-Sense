from __future__ import annotations

import mimetypes
import tempfile
from dataclasses import dataclass
from pathlib import Path

from typing import Optional

from fastapi import UploadFile


SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


@dataclass
class StoredUpload:
    source_name: str
    path: Path
    mime_type: str


async def store_uploads(files: Optional[list[UploadFile]], max_upload_mb: int) -> list[StoredUpload]:
    stored: list[StoredUpload] = []
    if not files:
        return stored

    max_bytes = max_upload_mb * 1024 * 1024
    temp_dir = Path(tempfile.mkdtemp(prefix="signsense_uploads_"))

    for upload in files:
        data = await upload.read()
        if len(data) > max_bytes:
            raise ValueError(f"{upload.filename or 'File'} is larger than {max_upload_mb} MB.")

        name = upload.filename or "uploaded-file"
        guessed_mime = upload.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        if guessed_mime not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported file type: {guessed_mime}.")

        suffix = Path(name).suffix or mimetypes.guess_extension(guessed_mime) or ".bin"
        path = temp_dir / f"{len(stored) + 1:02d}{suffix}"
        path.write_bytes(data)
        stored.append(StoredUpload(source_name=name, path=path, mime_type=guessed_mime))

    return stored


def cleanup_uploads(files: list[StoredUpload]) -> None:
    if not files:
        return
    temp_dir = files[0].path.parent
    for item in files:
        try:
            item.path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass
