import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore(ABC):
    @abstractmethod
    def put_artifact(self, run_id, local_path, artifact_path, producer, immutable=True):
        raise NotImplementedError

    @abstractmethod
    def get_artifact_path(self, run_id, artifact_path):
        raise NotImplementedError

    @abstractmethod
    def load_state(self, run_id):
        raise NotImplementedError

    @abstractmethod
    def atomic_update_state(self, run_id, state):
        raise NotImplementedError

    @abstractmethod
    def append_manifest_entry(self, run_id, entry):
        raise NotImplementedError

    @abstractmethod
    def load_manifest(self, run_id):
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("RESEARCH_ARTIFACT_ROOT", "research_runs")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run_root(self, run_id):
        return self.root / run_id

    def raw_root(self, run_id):
        return self.run_root(run_id) / "artifacts" / "raw"

    def state_root(self, run_id):
        return self.run_root(run_id) / "state"

    def ensure_run(self, run_id):
        self.raw_root(run_id).mkdir(parents=True, exist_ok=True)
        self.state_root(run_id).mkdir(parents=True, exist_ok=True)

    def put_artifact(self, run_id, local_path, artifact_path, producer, immutable=True):
        self.ensure_run(run_id)
        src = Path(local_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(local_path))
        checksum = sha256_file(src)
        destination = self.raw_root(run_id) / artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if immutable and destination.exists():
            existing = sha256_file(destination)
            if existing != checksum:
                raise ValueError(f"immutable artifact already exists with different checksum: {artifact_path}")
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(destination.parent), prefix=".artifact-", suffix=".tmp")
        os.close(tmp_fd)
        try:
            with open(src, "rb") as r, open(tmp_name, "wb") as w:
                for chunk in iter(lambda: r.read(1024 * 1024), b""):
                    w.write(chunk)
            os.replace(tmp_name, destination)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        entry = {
            "path": artifact_path,
            "producer": producer,
            "sha256": checksum,
            "bytes": destination.stat().st_size,
            "created_at": utc_now(),
            "immutable": immutable,
            "store": "local",
        }
        self.append_manifest_entry(run_id, entry)
        return entry

    def get_artifact_path(self, run_id, artifact_path):
        return str(self.raw_root(run_id) / artifact_path)

    def load_state(self, run_id):
        path = self.state_root(run_id) / "run_state.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def atomic_update_state(self, run_id, state):
        self.ensure_run(run_id)
        path = self.state_root(run_id) / "run_state.json"
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".state-", suffix=".tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_name, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def append_manifest_entry(self, run_id, entry):
        manifest = self.load_manifest(run_id)
        manifest["artifacts"].append(entry)
        self._write_manifest(run_id, manifest)

    def load_manifest(self, run_id):
        self.ensure_run(run_id)
        path = self.state_root(run_id) / "artifact_manifest.json"
        if not path.exists():
            return {"run_id": run_id, "artifacts": [], "schema_version": 1}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, run_id, manifest):
        path = self.state_root(run_id) / "artifact_manifest.json"
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_name, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def verify_manifest(self, run_id):
        failures = []
        for entry in self.load_manifest(run_id).get("artifacts", []):
            path = Path(self.get_artifact_path(run_id, entry["path"]))
            if not path.exists():
                failures.append({"path": entry["path"], "reason": "missing"})
            elif sha256_file(path) != entry.get("sha256"):
                failures.append({"path": entry["path"], "reason": "checksum_mismatch"})
        return failures


class S3CompatibleArtifactStore(ArtifactStore):
    def __init__(self, bucket=None, prefix=None, endpoint_url=None):
        self.bucket = bucket or os.environ.get("RESEARCH_S3_BUCKET")
        self.prefix = (prefix or os.environ.get("RESEARCH_S3_PREFIX", "")).strip("/")
        self.endpoint_url = endpoint_url or os.environ.get("RESEARCH_S3_ENDPOINT_URL")
        if not self.bucket:
            raise ValueError("RESEARCH_S3_BUCKET is required for S3-compatible storage")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install boto3 to use S3-compatible storage") from exc
        self.client = boto3.client("s3", endpoint_url=self.endpoint_url)

    def _key(self, run_id, name):
        parts = [p for p in [self.prefix, run_id, name] if p]
        return "/".join(parts)

    def put_artifact(self, run_id, local_path, artifact_path, producer, immutable=True):
        checksum = sha256_file(local_path)
        key = self._key(run_id, f"artifacts/raw/{artifact_path}")
        self.client.upload_file(local_path, self.bucket, key, ExtraArgs={"Metadata": {"sha256": checksum, "producer": producer}})
        entry = {"path": artifact_path, "producer": producer, "sha256": checksum, "created_at": utc_now(), "immutable": immutable, "store": "s3"}
        self.append_manifest_entry(run_id, entry)
        return entry

    def get_artifact_path(self, run_id, artifact_path):
        return f"s3://{self.bucket}/{self._key(run_id, f'artifacts/raw/{artifact_path}')}"

    def load_state(self, run_id):
        key = self._key(run_id, "state/run_state.json")
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return json.loads(obj["Body"].read().decode("utf-8"))

    def atomic_update_state(self, run_id, state):
        self.client.put_object(Bucket=self.bucket, Key=self._key(run_id, "state/run_state.json"), Body=json.dumps(state, indent=2, sort_keys=True).encode("utf-8"))

    def append_manifest_entry(self, run_id, entry):
        manifest = self.load_manifest(run_id)
        manifest["artifacts"].append(entry)
        self.client.put_object(Bucket=self.bucket, Key=self._key(run_id, "state/artifact_manifest.json"), Body=json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))

    def load_manifest(self, run_id):
        key = self._key(run_id, "state/artifact_manifest.json")
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            return {"run_id": run_id, "artifacts": [], "schema_version": 1}


def artifact_store_from_env():
    backend = os.environ.get("RESEARCH_ARTIFACT_STORE", "local").lower()
    if backend in ("local", "filesystem", "fs"):
        return LocalArtifactStore()
    if backend in ("s3", "r2", "minio", "s3-compatible"):
        return S3CompatibleArtifactStore()
    raise ValueError(f"unknown RESEARCH_ARTIFACT_STORE: {backend}")
