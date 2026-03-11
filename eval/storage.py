"""Storage backends for eval result persistence."""

import atexit
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

_pool = ThreadPoolExecutor(max_workers=2)
atexit.register(lambda: _pool.shutdown(wait=True))


def fire_and_forget(fn, *args):
    """Submit fn(*args) to background thread pool. Non-blocking."""
    _pool.submit(fn, *args)


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, local_path: Path) -> None: ...
    @abstractmethod
    def get(self, key: str, local_path: Path) -> bool: ...
    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...


class S3Storage(Storage):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, local_path: Path) -> None:
        self.s3.upload_file(str(local_path), self.bucket, self._key(key))

    def get(self, key: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._key(key), str(local_path))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        full = self._key(prefix) if prefix else (self.prefix or "")
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if self.prefix:
                    k = k.removeprefix(f"{self.prefix}/")
                keys.append(k)
        return keys
