"""Object-store access shared by the sink and the inspection tool."""

from __future__ import annotations

from recsys.config import Settings


def s3_filesystem(settings: Settings):
    """fsspec filesystem for the S3-compatible lake storage (MinIO-style endpoint)."""
    import fsspec

    return fsspec.filesystem(
        "s3",
        key=settings.s3_access_key,
        secret=settings.s3_secret_key,
        client_kwargs={"endpoint_url": settings.s3_endpoint},
    )


def lake_root(settings: Settings) -> str:
    return f"{settings.s3_bucket}/raw/user-events"
