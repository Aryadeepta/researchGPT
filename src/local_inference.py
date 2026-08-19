import json
import hashlib
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.llm_gateway import LLMProvider
from src.storage import sha256_file


TASK_THRESHOLDS = {
    "metadata_extraction": {"structured_output": 0.85},
    "candidate_question_generation": {"planning": 0.55, "structured_output": 0.75},
    "research_feasibility_analysis": {"feasibility_reasoning": 0.6, "structured_output": 0.75},
    "skill_code_generation": {"coding": 0.65, "structured_output": 0.8},
    "claim_adjudication": {"evidence_grounding": 0.8, "structured_output": 0.85},
    "adversarial_criticism": {"adversarial_criticism": 0.65},
}

CANONICAL_LLM_TASK_CLASSES = frozenset(TASK_THRESHOLDS)

STRUCTURED_SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"const": True}},
    "required": ["ok"],
    "additionalProperties": False,
}


def structured_generation_config(config, schema):
    if not isinstance(schema, dict):
        raise ValueError("structured generation requires a JSON schema")
    return {**config, "json_schema": schema, "reasoning": "off", "generation_mode": "STRUCTURED_GENERATION"}


class UnknownLLMTaskClass(Exception):
    pass


class NoEligibleLocalModel(RuntimeError):
    pass


class NoLocalProfileAvailable(RuntimeError):
    pass


class StructuredGenerationExhausted(RuntimeError):
    def __init__(self, message, attempts=None):
        super().__init__(message)
        self.attempts = attempts or []
        self.local_model = self.attempts[-1].get("actual_model") if self.attempts else None


class LocalRuntimeInfrastructureFailure(RuntimeError):
    def __init__(self, message, attempts=None):
        super().__init__(message)
        self.attempts = attempts or []
        self.local_model = self.attempts[-1].get("actual_model") if self.attempts else None


class LocalInferenceBackendIncompatible(LocalRuntimeInfrastructureFailure):
    pass


def probe_loopback_bind():
    result = {"status": "UNKNOWN", "host": "127.0.0.1", "port": 0, "errno": None, "message": None}
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        result.update({"status": "LOOPBACK_BIND_AVAILABLE", "port": sock.getsockname()[1]})
    except OSError as exc:
        result.update({"status": "LOOPBACK_BIND_UNAVAILABLE", "errno": exc.errno, "message": str(exc)})
    except Exception as exc:
        result.update({"status": "UNKNOWN", "message": str(exc)})
    finally:
        if sock is not None:
            sock.close()
    return result


class StructuredDecodingConfigurationFailure(RuntimeError):
    def __init__(self, message, attempts=None):
        super().__init__(message)
        self.attempts = attempts or []
        self.local_model = self.attempts[-1].get("actual_model") if self.attempts else None


QUALITY_BY_CLASS = {
    "CHEAP": ["metadata_extraction", "candidate_question_generation"],
    "STANDARD": ["candidate_question_generation", "research_feasibility_analysis"],
    "STRONG_LOCAL": ["claim_adjudication", "skill_code_generation", "adversarial_criticism"],
}


@dataclass
class QuantizationSpec:
    id: str
    bits_per_weight: float
    method: str
    quality_tier: str
    importance_aware: bool = False
    runtime_format: str = ""


@dataclass
class KVCacheSpec:
    id: str
    bits: int | None
    method: str
    turboquant_style: bool = False
    backend_supported: bool = False


class HardwareProbe:
    def probe(self):
        logical = os.cpu_count() or 1
        profile = {
            "cpu_model": platform.processor() or self._first_cpu_model(),
            "physical_cores": None,
            "logical_cores": logical,
            "instruction_sets": self._instruction_sets(),
            "system_ram_bytes": self._meminfo("MemTotal"),
            "available_ram_bytes": self._meminfo("MemAvailable"),
            "gpus": self._gpus(),
            "os": platform.platform(),
            "wsl": self._is_wsl(),
            "disk_available_bytes": shutil.disk_usage(os.getcwd()).free,
        }
        profile["supported_compute_backends"] = self._compute_backends(profile)
        return profile

    def persist(self, root):
        root = Path(root)
        path = root / "local_inference" / "hardware_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = self.probe()
        path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return profile

    def _first_cpu_model(self):
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            return ""
        return ""

    def _instruction_sets(self):
        try:
            text = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
        except Exception:
            text = ""
        return [flag for flag in ["avx", "avx2", "avx512", "fma", "sse4_2"] if flag in text]

    def _meminfo(self, key):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
        except Exception:
            return None
        return None

    def _gpus(self):
        gpus = []
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in out.stdout.splitlines():
                    name, _, mem = line.partition(",")
                    gpus.append({"name": name.strip(), "vendor": "nvidia", "vram_bytes": int(mem.strip()) * 1024 * 1024, "backend": "cuda"})
            except Exception:
                pass
        if not gpus and shutil.which("vulkaninfo"):
            gpus.append({"name": "vulkan-device", "vendor": "unknown", "vram_bytes": None, "backend": "vulkan"})
        return gpus

    def _is_wsl(self):
        try:
            return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
        except Exception:
            return False

    def _compute_backends(self, profile):
        backends = ["cpu"]
        for gpu in profile.get("gpus", []):
            if gpu.get("backend"):
                backends.append(gpu["backend"])
        if platform.system().lower() == "darwin":
            backends.append("metal")
        return sorted(set(backends))


class ModelRegistry:
    def __init__(self, root=None, size_limit_bytes=None):
        self.root = Path(root or os.environ.get("RESEARCH_LOCAL_MODEL_CACHE", ".local-model-cache"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "models.json"
        self.missing_models_path = self.root / "missing_models.json"
        self.size_limit_bytes = int(size_limit_bytes or os.environ.get("RESEARCH_LOCAL_MODEL_CACHE_LIMIT_BYTES", str(20 * 1024**3)))

    def load(self):
        if not self.index_path.exists():
            return {"models": []}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def save(self, index):
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def register(self, model):
        index = self.load()
        model = dict(model)
        model.setdefault("pinned", False)
        model.setdefault("last_used", 0)
        content_path = self._content_path(model)
        if content_path and content_path.exists():
            model["size_bytes"] = content_path.stat().st_size
            model["sha256"] = sha256_file(content_path)
        index["models"] = [m for m in index["models"] if m.get("id") != model["id"]]
        index["models"].append(model)
        self.save(index)
        return model

    def discover_from_env(self):
        models = []
        missing_environment = []
        missing_configured = []
        for raw in self._configured_model_paths():
            path = Path(raw)
            if path.exists():
                quant = infer_quantization_from_name(path.name)
                models.append(self.register({
                    "id": path.stem,
                    "path": str(path),
                    "format": path.suffix.lstrip(".").lower(),
                    "quantization": quant.id,
                    "bits_per_weight": quant.bits_per_weight,
                    "estimated_size_bytes": path.stat().st_size,
                    "quality_hint": quant.quality_tier,
                    "source_kind": "environment_override",
                }))
            else:
                missing_environment.append({"path": raw, "reason": "missing", "source_kind": "environment_override"})
        for spec in self.configured_model_specs():
            path = spec.get("path")
            if path and Path(path).exists():
                models.append(self.register(self._model_from_spec(spec, Path(path))))
            elif path:
                cached = self.find_cached_hf_model(spec)
                if cached:
                    models.append(self.register(self._model_from_spec(spec, cached["snapshot_path"], cached)))
                else:
                    missing_configured.append({"profile": spec.get("profile"), "path": path, "reason": "missing", "source_kind": "configured_registry"})
            else:
                cached = self.find_cached_hf_model(spec)
                if cached:
                    models.append(self.register(self._model_from_spec(spec, cached["snapshot_path"], cached)))
                else:
                    missing_configured.append({
                        "id": spec.get("id"),
                        "profile": spec.get("profile"),
                        "source": spec.get("source"),
                        "quantization": spec.get("quantization"),
                        "reason": "configured_but_no_local_path",
                        "source_kind": "configured_registry",
                    })
        missing = missing_environment + missing_configured
        if missing:
            self.missing_models_path.write_text(json.dumps({
                "missing_models": missing,
                "missing_environment_models": missing_environment,
                "missing_configured_models": missing_configured,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif self.missing_models_path.exists():
            self.missing_models_path.unlink()
        return models

    def _content_path(self, model):
        candidate = model.get("resolved_path") or model.get("path")
        if not candidate:
            return None
        path = Path(candidate)
        try:
            return path.resolve(strict=True)
        except FileNotFoundError:
            return path if path.exists() else None

    def _configured_model_paths(self):
        return list(filter(None, os.environ.get("RESEARCH_LOCAL_MODELS", "").split(os.pathsep)))

    def configured_model_specs(self):
        spec_path = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
        if not spec_path:
            default = Path("config/local_models.json")
            spec_path = str(default) if default.exists() else ""
        if not spec_path:
            return []
        path = Path(spec_path)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", data if isinstance(data, list) else [])

    def hf_cache_roots(self):
        roots = []
        for raw in [
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ]:
            if raw and raw not in roots:
                roots.append(raw)
        return [Path(root) for root in roots]

    def specs_by_profile(self):
        return {(spec.get("profile") or spec.get("task_tier", "")).lower(): spec for spec in self.configured_model_specs()}

    def _model_from_spec(self, spec, path, cache_info=None):
        path = Path(path)
        content_path = Path(cache_info["resolved_path"]) if cache_info else path
        quant = infer_quantization_from_name(path.name)
        source = dict(spec.get("source", {}))
        if cache_info:
            source.setdefault("repo", cache_info.get("repo"))
            source.setdefault("revision", cache_info.get("revision"))
        symlink_backed = bool(cache_info and cache_info.get("symlink_backed")) or path.is_symlink()
        if not cache_info and symlink_backed:
            try:
                content_path = path.resolve(strict=True)
            except FileNotFoundError:
                content_path = path
        snapshot_path = str(path) if cache_info or symlink_backed else None
        cache_revision = cache_info.get("revision") if cache_info else path.parent.name if path.parent.parent.name == "snapshots" else None
        return {
            "id": spec.get("id") or path.stem,
            "profile": spec.get("profile") or spec.get("task_tier"),
            "path": str(path),
            "resolved_path": str(content_path) if cache_info else str(content_path.resolve()) if content_path.exists() else str(content_path),
            "snapshot_path": snapshot_path,
            "symlink_backed": symlink_backed,
            "cache_revision": cache_revision,
            "format": path.suffix.lstrip(".").lower(),
            "source": source,
            "quantization": spec.get("quantization") or quant.id,
            "bits_per_weight": spec.get("bits_per_weight") or quant.bits_per_weight,
            "estimated_size_bytes": content_path.stat().st_size if content_path.exists() else None,
            "quality_hint": spec.get("quality_hint") or quant.quality_tier,
            "task_tier": spec.get("task_tier"),
            "source_kind": "configured_registry",
        }

    def find_cached_hf_model(self, spec):
        source = spec.get("source", {})
        if source.get("type") != "huggingface" or not source.get("repo"):
            return None
        repo = source["repo"]
        repo_dir_name = "models--" + repo.replace("/", "--")
        wanted_file = source.get("filename") or source.get("file")
        wanted_quant = source.get("quant") or spec.get("quantization")
        matches = []
        for root in self.hf_cache_roots():
            repo_dir = root / repo_dir_name
            snapshots = repo_dir / "snapshots"
            if not snapshots.exists():
                continue
            for entry in snapshots.glob("*/*.gguf"):
                if not self._matches_hf_spec(entry, wanted_file, wanted_quant):
                    continue
                resolved = self._safe_resolve_hf_cache_entry(entry, root, repo_dir)
                if not resolved:
                    continue
                matches.append({
                    "snapshot_path": entry,
                    "resolved_path": resolved,
                    "symlink_backed": entry.is_symlink(),
                    "repo": repo,
                    "revision": entry.parent.name,
                })
        if not matches:
            return None
        return sorted(matches, key=lambda item: str(item["snapshot_path"]))[0]

    def _matches_hf_spec(self, entry, wanted_file=None, wanted_quant=None):
        if wanted_file and entry.name != wanted_file:
            return False
        if wanted_quant:
            normalized_name = normalize_quant_token(entry.name)
            normalized_quant = normalize_quant_token(wanted_quant)
            if normalized_quant not in normalized_name:
                return False
        return True

    def _safe_resolve_hf_cache_entry(self, entry, cache_root, repo_dir):
        if not entry.exists():
            return None
        try:
            resolved = entry.resolve(strict=True)
        except FileNotFoundError:
            return None
        if not resolved.is_file():
            return None
        allowed_roots = [cache_root.resolve(), repo_dir.resolve()]
        try:
            if not any(resolved.is_relative_to(root) for root in allowed_roots):
                return None
        except AttributeError:
            if not any(str(resolved).startswith(str(root) + os.sep) or resolved == root for root in allowed_roots):
                return None
        return resolved

    def missing_from_env(self):
        if self.missing_models_path.exists():
            return json.loads(self.missing_models_path.read_text(encoding="utf-8")).get("missing_models", [])
        return []

    def missing_by_source(self):
        if not self.missing_models_path.exists():
            return {"missing_environment_models": [], "missing_configured_models": []}
        data = json.loads(self.missing_models_path.read_text(encoding="utf-8"))
        return {
            "missing_environment_models": data.get("missing_environment_models", []),
            "missing_configured_models": data.get("missing_configured_models", []),
        }

    def install_profile(self, profile, runtime=None):
        spec = self.specs_by_profile().get(profile.lower())
        if not spec:
            raise ValueError(f"no local model spec configured for profile: {profile}")
        path = spec.get("path")
        if path and Path(path).exists():
            return {"status": "already_present", "model": self.register(self._model_from_spec(spec, Path(path)))}
        cached = self.find_cached_hf_model(spec)
        if cached:
            return {"status": "already_present", "model": self.register(self._model_from_spec(spec, cached["snapshot_path"], cached))}
        source = spec.get("source", {})
        if source.get("type") != "huggingface":
            raise ValueError("only explicit Hugging Face model specs are currently installable")
        runtime = runtime or LlamaCppRuntime()
        if not runtime.executable:
            raise RuntimeError("llama.cpp executable is required for model install")
        quant = source.get("quant") or spec.get("quantization")
        hf_arg = f"{source['repo']}:{quant}" if quant else source["repo"]
        cmd = runtime.download_command(["-hf", hf_arg])
        source_file = source.get("file") or source.get("filename")
        if source_file:
            cmd.extend(["--hf-file", source_file])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("RESEARCH_LOCAL_MODEL_DOWNLOAD_TIMEOUT", "1800")))
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
        record = {
            "id": spec.get("id") or f"{profile}-{quant}",
            "profile": spec.get("profile") or profile,
            "runtime": "llama.cpp",
            "source": source,
            "quantization": spec.get("quantization") or quant,
            "bits_per_weight": spec.get("bits_per_weight") or infer_quantization_from_name(str(quant)).bits_per_weight,
            "task_tier": spec.get("task_tier"),
            "install_command": cmd,
            "install_stdout": result.stdout[-4000:],
            "install_stderr": result.stderr[-4000:],
        }
        installed_path = self._extract_downloaded_path(result.stdout + "\n" + result.stderr)
        if installed_path and Path(installed_path).exists():
            record.update(self._model_from_spec({**spec, "path": installed_path}, Path(installed_path)))
        elif path:
            record["path"] = path
        return {"status": "installed", "model": self.register(record)}

    def _extract_downloaded_path(self, output):
        for token in output.replace("\n", " ").split():
            cleaned = token.strip("'\"")
            if cleaned.endswith(".gguf") and Path(cleaned).exists():
                return cleaned
        return None

    def candidates(self):
        return self.discover_from_env() or self.load().get("models", [])

    def enforce_limit(self):
        index = self.load()
        total = sum(int(m.get("size_bytes") or m.get("estimated_size_bytes") or 0) for m in index["models"])
        removed = []
        if total <= self.size_limit_bytes:
            return removed
        models = sorted(index["models"], key=lambda m: (m.get("pinned", False), m.get("last_used", 0)))
        kept = []
        for model in models:
            size = int(model.get("size_bytes") or model.get("estimated_size_bytes") or 0)
            if total > self.size_limit_bytes and not model.get("pinned") and model.get("managed_cache_file"):
                path = Path(model["managed_cache_file"])
                if path.exists() and path.is_file():
                    path.unlink()
                total -= size
                removed.append(model["id"])
            else:
                kept.append(model)
        index["models"] = kept
        self.save(index)
        return removed


def infer_quantization_from_name(name):
    lower = name.lower()
    table = [
        ("iq", 3, "importance_aware"),
        ("q8", 8, "high"),
        ("q6", 6, "medium_high"),
        ("q5", 5, "medium"),
        ("q4", 4, "balanced"),
        ("q3", 3, "aggressive"),
        ("q2", 2, "extreme"),
    ]
    for token, bits, tier in table:
        if token in lower:
            return QuantizationSpec(token.upper(), bits, token.upper(), tier, importance_aware=token == "iq", runtime_format=token.upper())
    return QuantizationSpec("UNKNOWN", 8, "unknown", "unknown")


def normalize_quant_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class QuantizationPolicy:
    def __init__(self, mode=None):
        self.mode = mode or os.environ.get("RESEARCH_LOCAL_OPTIMIZATION", "balanced")

    def candidates(self):
        if self.mode == "aggressive":
            return [
                QuantizationSpec("Q2", 2, "low_bit", "extreme"),
                QuantizationSpec("Q3", 3, "low_bit", "aggressive"),
                QuantizationSpec("Q4", 4, "low_bit", "balanced"),
                QuantizationSpec("Q5", 5, "low_bit", "medium"),
            ]
        return [
            QuantizationSpec("Q4", 4, "low_bit", "balanced"),
            QuantizationSpec("Q5", 5, "low_bit", "medium"),
            QuantizationSpec("Q6", 6, "low_bit", "medium_high"),
            QuantizationSpec("Q8", 8, "low_bit", "high"),
        ]


class KVCachePolicy:
    def __init__(self, mode=None):
        self.mode = mode or os.environ.get("RESEARCH_LOCAL_KV_CACHE", "auto")

    def candidates(self, runtime_caps=None):
        runtime_caps = runtime_caps or {}
        modes = [KVCacheSpec("fp16", 16, "full_precision", backend_supported=True)]
        if runtime_caps.get("kv_quantization"):
            supported = runtime_caps.get("supported_kv_cache_ids") or ["q8", "q4", "q3"]
            for kv_id, bits in [("q8", 8), ("q4", 4), ("q3", 3)]:
                if kv_id in supported:
                    modes.append(KVCacheSpec(kv_id, bits, "quantized", backend_supported=True))
        if runtime_caps.get("turboquant"):
            modes.append(KVCacheSpec("turboquant", None, "turboquant", turboquant_style=True, backend_supported=True))
        if self.mode == "aggressive":
            return sorted(modes, key=lambda m: 99 if m.bits is None else m.bits)
        return modes


class LocalRuntime:
    name = "abstract"

    def discover(self):
        return {"available": False, "capabilities": {}}

    def supports(self, model):
        return False

    def start(self, config):
        raise NotImplementedError

    def stop(self):
        return None

    def generate(self, prompt, config):
        raise NotImplementedError

    def benchmark(self, config):
        raise NotImplementedError


class LlamaCppRuntime(LocalRuntime):
    name = "llama.cpp"

    def __init__(self, executable=None, invocation_mode=None, loopback_probe=None):
        self.loopback_probe = loopback_probe or probe_loopback_bind
        self.executable, self.invocation_mode, self.discovery_source = self._discover_binary(executable, invocation_mode)
        self.loopback_diagnostics = self.loopback_probe()
        self.requires_local_socket = self.invocation_mode == "UNIFIED_CLI_ROUTER"
        self.backend_compatibility = ("BACKEND_INCOMPATIBLE_WITH_ENVIRONMENT"
            if self.requires_local_socket and self.loopback_diagnostics.get("status") == "LOOPBACK_BIND_UNAVAILABLE"
            else "COMPATIBLE" if self.executable else "UNAVAILABLE")
        self.backend_id = f"llama.cpp:{self.invocation_mode or 'UNAVAILABLE'}:{Path(self.executable).name if self.executable else 'none'}"

    def _discover_binary(self, explicit=None, explicit_mode=None):
        if explicit:
            path = shutil.which(explicit) or (explicit if Path(explicit).exists() else None)
            mode = self._normalize_invocation_mode(explicit_mode) or self._infer_invocation_mode(path or explicit)
            return path, mode, "CONSTRUCTOR_ARGUMENT"
        configured_direct = os.environ.get("LLAMA_CPP_CLI_BIN")
        if configured_direct:
            path = shutil.which(configured_direct) or (configured_direct if Path(configured_direct).exists() else None)
            return path, "DIRECT_CLI", "LLAMA_CPP_CLI_BIN"
        path = shutil.which("llama-cli")
        if path:
            return path, "DIRECT_CLI", "PATH_LLAMA_CLI"
        configured = os.environ.get("LLAMA_CPP_BIN")
        if configured:
            path = shutil.which(configured) or (configured if Path(configured).exists() else None)
            if path:
                sibling = Path(path).with_name("llama-cli")
                if sibling.exists() and os.access(sibling, os.X_OK):
                    return str(sibling), "DIRECT_CLI", "SIBLING_OF_LLAMA_CPP_BIN"
            mode = self._normalize_invocation_mode(explicit_mode) or self._infer_invocation_mode(path or configured)
            return path, mode, "LLAMA_CPP_BIN"
        for candidate in ("llama", "main"):
            path = shutil.which(candidate)
            if path:
                return path, self._infer_invocation_mode(path), f"PATH_{candidate.upper()}"
        return None, None, "NOT_FOUND"

    def _normalize_invocation_mode(self, mode):
        return {"unified": "UNIFIED_CLI_ROUTER", "legacy": "DIRECT_CLI",
                "UNIFIED_CLI_ROUTER": "UNIFIED_CLI_ROUTER", "DIRECT_CLI": "DIRECT_CLI"}.get(mode)

    def _infer_invocation_mode(self, path):
        name = Path(path).name
        if name == "llama":
            return "UNIFIED_CLI_ROUTER"
        return "DIRECT_CLI"

    def discover(self):
        available = bool(self.executable)
        compatible = self.backend_compatibility == "COMPATIBLE"
        server_available = self._server_available() if compatible else False
        caps = {
            "gguf": True,
            "gpu_offload": True,
            "hybrid_cpu_gpu": True,
            "mmap": True,
            "kv_quantization": True,
            "supported_kv_cache_ids": ["q8", "q4"],
            "json_schema_constraint": self._supports_json_schema() if compatible else False,
            "turboquant": False,
            "openai_compatible_server": server_available,
            "speculative_decoding": True,
        }
        return {
            "runtime": self.name,
            "available": available,
            "backend_id": self.backend_id,
            "executable": self.executable,
            "invocation_mode": self.invocation_mode,
            "executable_version": self._executable_version() if compatible else "NOT_PROBED_INCOMPATIBLE_BACKEND",
            "requires_local_socket": self.requires_local_socket,
            "loopback_bind_status": self.loopback_diagnostics.get("status"),
            "loopback_bind_diagnostics": self.loopback_diagnostics,
            "backend_compatibility": self.backend_compatibility,
            "discovery_source": self.discovery_source,
            "server_executable": shutil.which("llama-server") if self.invocation_mode == "DIRECT_CLI" else self.executable,
            "capabilities": caps,
        }

    def _executable_version(self):
        if not self.executable:
            return None
        try:
            result = subprocess.run([self.executable, "--version"], capture_output=True, text=True, timeout=3)
            text = (result.stdout or result.stderr or "").strip().splitlines()
            return text[0][:240] if text else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _server_available(self):
        if not self.executable:
            return False
        if self.invocation_mode == "UNIFIED_CLI_ROUTER":
            return self._unified_supports("serve")
        return bool(shutil.which("llama-server"))

    def _unified_supports(self, subcommand):
        if not self.executable or self.invocation_mode != "UNIFIED_CLI_ROUTER":
            return False
        try:
            result = subprocess.run([self.executable, subcommand, "--help"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0 or "usage" in (result.stdout + result.stderr).lower()
        except Exception:
            return False

    def _supports_json_schema(self):
        if not self.executable:
            return False
        try:
            result = subprocess.run(self.cli_command(["--help"]), capture_output=True, text=True, timeout=5)
            return "--json-schema" in (result.stdout + result.stderr)
        except Exception:
            return False

    def supports(self, model):
        return (str(model.get("format", "")).lower() == "gguf" and bool(self.executable)
                and self.backend_compatibility == "COMPATIBLE")

    def cli_command(self, args):
        if not self.executable:
            raise RuntimeError("llama.cpp executable not found")
        if self.invocation_mode == "UNIFIED_CLI_ROUTER":
            return [self.executable, "cli", *args]
        return [self.executable, *args]

    def serve_command(self, args):
        if not self.executable:
            raise RuntimeError("llama.cpp executable not found")
        if self.invocation_mode == "UNIFIED_CLI_ROUTER":
            return [self.executable, "serve", *args]
        server = shutil.which("llama-server")
        if not server:
            raise RuntimeError("llama-server executable not found")
        return [server, *args]

    def download_command(self, args):
        if not self.executable:
            raise RuntimeError("llama.cpp executable not found")
        if self.invocation_mode == "UNIFIED_CLI_ROUTER":
            return [self.executable, "download", *args]
        return [self.executable, *args]

    def cache_list_command(self):
        return self.cli_command(["--cache-list"])

    def generate(self, prompt, config):
        diagnostic = self.run_cli(prompt, config)
        if diagnostic["status"] not in ("SUCCESS",):
            exc = RuntimeError(diagnostic.get("stderr")[-1000:] or diagnostic.get("stdout")[-1000:] or diagnostic["status"])
            setattr(exc, "diagnostic", diagnostic)
            raise exc
        return {
            "text": diagnostic["parsed_assistant_output"],
            "duration_seconds": diagnostic["wall_clock_duration_seconds"],
            "model": config["id"],
            "diagnostic": diagnostic,
        }

    def run_cli(self, prompt, config):
        if not self.executable:
            raise RuntimeError("llama.cpp executable not found")
        if self.backend_compatibility != "COMPATIBLE":
            diagnostic = {"status": "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE", "command": None,
                "backend": self.discover(), "resource_diagnostics": {"backend_compatibility": self.backend_compatibility,
                    "loopback_bind_status": self.loopback_diagnostics.get("status")}}
            exc = LocalInferenceBackendIncompatible(
                "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE unified router requires unavailable loopback binding")
            exc.diagnostic = diagnostic
            raise exc
        prompt_for_cli = render_programmatic_llama_prompt(prompt) if config.get("json_schema") is not None else prompt
        args = [
            "-m",
            config["model"]["path"],
            "-p",
            prompt_for_cli,
            "-n",
            str(config.get("max_tokens", 256)),
            "-c",
            str(config.get("context", 2048)),
            "-t",
            str(config.get("threads", os.environ.get("RESEARCH_LOCAL_THREADS", "2"))),
            "-tb",
            str(config.get("threads_batch", config.get("threads", os.environ.get("RESEARCH_LOCAL_THREADS", "2")))),
            "-st",
            "--simple-io",
            "--no-display-prompt",
            "--reasoning",
            str(config.get("reasoning", os.environ.get("RESEARCH_LOCAL_REASONING", "off"))),
        ]
        if config.get("json_schema") is not None:
            args.append("--no-jinja")
            args.extend(["--json-schema", json.dumps(config["json_schema"], sort_keys=True)])
        if config.get("gpu_layers") is not None:
            args.extend(["-ngl", str(config["gpu_layers"])])
        kv_type = self._llama_kv_cache_type(config.get("kv_cache", {}).get("id"))
        if kv_type:
            args.extend(["--cache-type-k", kv_type, "--cache-type-v", kv_type])
        cmd = self.cli_command(args)
        start = time.time()
        stdout = ""
        stderr = ""
        exit_code = None
        status = "SUCCESS"
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("RESEARCH_LOCAL_MAX_RUNTIME_SECONDS", "120")))
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.returncode
            if result.returncode != 0:
                status = "EXECUTION_FAILED"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            status = "TIMEOUT"
        duration = time.time() - start
        if is_structured_decoding_error(stdout, stderr):
            status = "STRUCTURED_DECODING_CONFIGURATION_FAILURE"
        extraction = {
            "assistant_text": "",
            "parsed_response": "",
            "strategy": "skipped_structured_decoding_failure",
            "status": status,
            "diagnostics": {},
        }
        if status != "STRUCTURED_DECODING_CONFIGURATION_FAILURE":
            extraction = extract_llama_assistant_response(stdout, prompt_for_cli, structured=bool(config.get("json_schema") is not None))
        parsed = extraction.get("parsed_response", "")
        if status == "SUCCESS" and extraction.get("status") != "SUCCESS":
            status = extraction.get("status") or "MODEL_OUTPUT_EXTRACTION_FAILED"
        if status == "SUCCESS" and not parsed:
            status = "EMPTY_OUTPUT"
        parse_status = "SUCCESS" if status == "SUCCESS" and parsed else status
        parsed_json = parse_json_object(parsed)
        return {
            "command": cmd,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "wall_clock_duration_seconds": duration,
            "assistant_generation_text": extraction.get("assistant_text", ""),
            "isolated_assistant_text": extraction.get("assistant_text", ""),
            "parsed_assistant_output": parsed,
            "extraction_strategy": extraction.get("strategy"),
            "extraction_diagnostics": extraction.get("diagnostics", {}),
            "parse_status": parse_status,
            "json_parse_mode": json_parse_mode(parsed, parsed_json),
            "status": status,
            "timing_data": parse_llama_timing(stdout + "\n" + stderr),
            "model": config["id"],
            "resource_diagnostics": {
                "backend_id": self.backend_id,
                "invocation_mode": self.invocation_mode,
                "executable": self.executable,
                "requires_local_socket": self.requires_local_socket,
                "loopback_bind_status": self.loopback_diagnostics.get("status"),
                "backend_compatibility": self.backend_compatibility,
                "discovery_source": self.discovery_source,
                "unified_cli_uses_internal_localhost_router": self.invocation_mode == "UNIFIED_CLI_ROUTER",
                "process_cleanup": "subprocess_run_waited; timeout path killed and reaped by subprocess.run",
            },
        }

    def _llama_kv_cache_type(self, kv_id):
        return {"q8": "q8_0", "q4": "q4_0"}.get(kv_id)

    def benchmark(self, config):
        start = time.time()
        result = self.generate("Return JSON: {\"ok\": true}", {**config, "max_tokens": 32})
        duration = max(time.time() - start, 0.001)
        return {"tokens_per_second": 32 / duration, "duration_seconds": result["duration_seconds"]}


class RuntimeRegistry:
    def __init__(self, runtimes=None):
        self.runtimes = runtimes or [LlamaCppRuntime()]

    def discover(self):
        return [runtime.discover() for runtime in self.runtimes]

    def first_supporting(self, model):
        for runtime in self.runtimes:
            if runtime.supports(model):
                return runtime
        return None

    def incompatibility_diagnostics(self):
        return [runtime.discover() for runtime in self.runtimes
                if getattr(runtime, "executable", None) and getattr(runtime, "backend_compatibility", "COMPATIBLE") != "COMPATIBLE"]


def estimate_fit(model, hardware, context=2048, kv_cache=None, ram_fraction=None, vram_fraction=None):
    ram_fraction = float(ram_fraction or os.environ.get("RESEARCH_LOCAL_MAX_RAM_FRACTION", "0.55"))
    vram_fraction = float(vram_fraction or os.environ.get("RESEARCH_LOCAL_MAX_VRAM_FRACTION", "0.75"))
    available_ram = int(hardware.get("available_ram_bytes") or hardware.get("system_ram_bytes") or 0)
    usable_ram = int(available_ram * ram_fraction)
    model_size = int(model.get("estimated_size_bytes") or model.get("size_bytes") or 0)
    kv_bits = (kv_cache or {}).get("bits") or 16
    hidden_guess = int(model.get("hidden_size", 4096))
    layers_guess = int(model.get("layers", 24))
    kv_estimate = int(context * hidden_guess * layers_guess * 2 * kv_bits / 8)
    overhead = int(0.6 * 1024**3)
    ram_estimate = model_size + kv_estimate + overhead
    gpu = next((g for g in hardware.get("gpus", []) if g.get("vram_bytes")), None)
    usable_vram = int(gpu["vram_bytes"] * vram_fraction) if gpu else 0
    return {
        "fits_ram": ram_estimate <= usable_ram if usable_ram else False,
        "fits_vram": model_size <= usable_vram if usable_vram else False,
        "ram_estimate_bytes": ram_estimate,
        "vram_estimate_bytes": min(model_size, usable_vram) if usable_vram else 0,
        "usable_ram_bytes": usable_ram,
        "usable_vram_bytes": usable_vram,
        "context": context,
        "kv_bits": kv_bits,
    }


class ContextReducer:
    def __init__(self, max_context=None):
        self.max_context = int(max_context or os.environ.get("RESEARCH_LOCAL_MAX_CONTEXT", "2048"))

    def reduce(self, context):
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception:
                pass
        if isinstance(context, dict):
            selected = {
                key: context.get(key)
                for key in [
                    "task",
                    "topic",
                    "research_question",
                    "candidate_questions",
                    "required_schema",
                    "evaluation_dimensions",
                    "relevant_claims",
                    "literature",
                    "literature_excerpts",
                    "artifact_summaries",
                    "decision_summary",
                ]
                if key in context
            }
            for lit_key in ("literature", "literature_excerpts"):
                if isinstance(selected.get(lit_key), list):
                    selected[lit_key] = [compact_literature_item(item) for item in selected[lit_key][:4]]
            text = json.dumps(selected, sort_keys=True)
        else:
            text = str(context)
        words = text.split()
        max_words = max(1, int(self.max_context * float(os.environ.get("RESEARCH_LOCAL_CONTEXT_WORD_FRACTION", "0.45"))))
        return " ".join(words[:max_words])

    def persist_summary(self, root, name, summary):
        path = Path(root) / "local_inference" / "context" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(path)


class CapabilityBenchmark:
    def synthetic_scores(self, config):
        quant_bits = float(config.get("quantization", {}).get("bits_per_weight") or config.get("model", {}).get("bits_per_weight") or 8)
        context = int(config.get("context", 2048))
        base = min(1.0, max(0.2, quant_bits / 6.0))
        context_bonus = min(0.15, context / 32768)
        coding_penalty = 0.12 if quant_bits <= 3 else 0.0
        return {
            "structured_output": round(min(0.98, base + context_bonus), 3),
            "planning": round(min(0.95, base * 0.85 + context_bonus), 3),
            "coding": round(max(0.05, min(0.92, base * 0.78 + context_bonus - coding_penalty)), 3),
            "evidence_grounding": round(min(0.94, base * 0.82 + context_bonus), 3),
            "feasibility_reasoning": round(min(0.92, base * 0.84 + context_bonus), 3),
            "adversarial_criticism": round(min(0.9, base * 0.8 + context_bonus), 3),
        }

    def eligible_tasks(self, capabilities):
        eligible = []
        for task, thresholds in TASK_THRESHOLDS.items():
            if all(capabilities.get(metric, 0) >= threshold for metric, threshold in thresholds.items()):
                eligible.append(task)
        return eligible

    def persist(self, root, config, benchmark):
        path = Path(root) / "local_inference" / "benchmarks" / f"{config['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(path)

    def measured_benchmark(self, runtime, config, threads=2):
        tasks = {
            "structured_output": 'Return exactly one JSON object and no other text: {"ok": true}',
            "planning": 'Return one JSON object with key "steps" containing three short generic research planning steps.',
            "coding": 'Return one JSON object with key "code" containing a short Python program that writes out.json.',
            "evidence_grounding": 'Given evidence [{"id":"A","value":2}], return one JSON object {"claim_supported": true, "evidence_id": "A"}.',
            "feasibility_reasoning": (
                'Given research question "Can a small local script compare two public metadata records?", '
                'resources {"python": true, "network": false, "max_runtime_minutes": 5}, '
                'and required evidence ["raw records", "comparison output"], return one JSON object with keys '
                '"verdict", "evidence_requirements", "resource_constraints", and "validation_plan".'
            ),
            "adversarial_criticism": 'Return one JSON object with key "flaws" listing one generic way a claim could be falsified.',
        }
        raw_outputs = {}
        scores = {}
        started = time.time()
        for metric, prompt in tasks.items():
            trial_config = {**config, "threads": threads, "max_tokens": 96}
            diagnostic = self._run_task(runtime, prompt, trial_config)
            parsed_json = parse_json_object(diagnostic.get("parsed_assistant_output", ""))
            if diagnostic["status"] == "SUCCESS" and parsed_json is None:
                diagnostic["status"] = "PARSE_FAILED"
                diagnostic["parse_status"] = "PARSE_FAILED"
            elif diagnostic["status"] == "SUCCESS":
                diagnostic["parse_status"] = "SUCCESS"
                scores[metric] = self._score_metric(metric, parsed_json)
                diagnostic["failure_class"] = "MODEL_CAPABILITY_FAILURE" if scores[metric] == 0.0 else None
            diagnostic.setdefault("failure_class", classify_local_failure(diagnostic))
            raw_outputs[metric] = diagnostic
        duration = time.time() - started
        timing = aggregate_timing(raw_outputs.values())
        return {
            "capabilities": scores,
            "measured_capability": scores,
            "eligible_task_classes": self.eligible_tasks(scores),
            "raw_outputs": raw_outputs,
            "wall_clock_duration_seconds": duration,
            "prompt_tokens_per_second": timing.get("prompt_tokens_per_second"),
            "generation_tokens_per_second": timing.get("generation_tokens_per_second"),
            "threads": threads,
        }

    def diagnostic_smoke(self, runtime, config, threads=2):
        trial_config = structured_generation_config(
            {**config, "threads": threads, "max_tokens": 32}, STRUCTURED_SMOKE_SCHEMA)
        diagnostic = self._run_task(runtime, 'Return exactly {"ok":true}', trial_config)
        parsed_json = parse_json_object(diagnostic.get("parsed_assistant_output", ""))
        schema_errors = validate_json_schema_subset(parsed_json, STRUCTURED_SMOKE_SCHEMA) if parsed_json is not None else []
        parse_mode = json_parse_mode(diagnostic.get("parsed_assistant_output", ""), parsed_json)
        if (diagnostic["status"] == "SUCCESS" and parsed_json == {"ok": True}
                and not schema_errors and parse_mode == "STRICT_JSON_ONLY"):
            diagnostic["parse_status"] = "SUCCESS"
            diagnostic["json_parse_mode"] = parse_mode
            diagnostic["failure_class"] = None
        elif diagnostic["status"] == "SUCCESS" and parsed_json is None:
            diagnostic["status"] = "PARSE_FAILED"
            diagnostic["parse_status"] = "PARSE_FAILED"
            diagnostic["failure_class"] = "MODEL_OUTPUT_FAILURE"
        elif diagnostic["status"] == "SUCCESS":
            diagnostic["status"] = "PARSE_FAILED"
            diagnostic["parse_status"] = "PARSE_FAILED"
            diagnostic["failure_class"] = "MODEL_OUTPUT_FAILURE"
        else:
            diagnostic["failure_class"] = classify_local_failure(diagnostic) or "MODEL_OUTPUT_FAILURE"
        diagnostic["parsed_json"] = parsed_json
        diagnostic["schema_errors"] = schema_errors
        diagnostic["structured_decoding"] = True
        diagnostic["reasoning_disabled"] = trial_config.get("reasoning") == "off"
        diagnostic["jinja_disabled"] = bool(trial_config.get("json_schema"))
        diagnostic["json_schema_enabled"] = bool(trial_config.get("json_schema"))
        diagnostic["schema_hash"] = hashlib.sha256(
            json.dumps(STRUCTURED_SMOKE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return diagnostic

    def _run_task(self, runtime, prompt, trial_config):
        if hasattr(runtime, "run_cli"):
            return runtime.run_cli(prompt, trial_config)
        try:
            result = runtime.generate(prompt, trial_config)
            text = result.get("text", "")
            return {
                "command": result.get("command"),
                "exit_code": result.get("exit_code", 0),
                "stdout": result.get("stdout", text),
                "stderr": result.get("stderr", ""),
                "wall_clock_duration_seconds": result.get("duration_seconds"),
                "parsed_assistant_output": text,
                "parse_status": "SUCCESS" if text else "EMPTY_OUTPUT",
                "status": "SUCCESS" if text else "EMPTY_OUTPUT",
                "failure_class": None if text else "MODEL_OUTPUT_FAILURE",
                "json_parse_mode": json_parse_mode(text, parse_json_object(text)),
                "timing_data": result.get("timing_data", {}),
            }
        except Exception as exc:
            return {
                "command": trial_config.get("command"),
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
                "wall_clock_duration_seconds": None,
                "parsed_assistant_output": "",
                "parse_status": "EXECUTION_FAILED",
                "status": "EXECUTION_FAILED",
                "failure_class": "RUNTIME_ENVIRONMENT_FAILURE",
                "json_parse_mode": "NO_JSON",
                "timing_data": {},
            }

    def _score_metric(self, metric, parsed_json):
        if metric == "structured_output":
            return 1.0 if parsed_json == {"ok": True} else 0.0
        if metric == "planning":
            return 1.0 if isinstance(parsed_json.get("steps"), list) and len(parsed_json["steps"]) >= 3 else 0.0
        if metric == "coding":
            return 1.0 if isinstance(parsed_json.get("code"), str) and "out.json" in parsed_json["code"] else 0.0
        if metric == "evidence_grounding":
            return 1.0 if parsed_json.get("claim_supported") is True and parsed_json.get("evidence_id") == "A" else 0.0
        if metric == "feasibility_reasoning":
            return 1.0 if (
                parsed_json.get("verdict") in {"FEASIBLE", "PARTIAL", "BLOCKED"}
                and isinstance(parsed_json.get("evidence_requirements"), list)
                and parsed_json.get("evidence_requirements")
                and isinstance(parsed_json.get("resource_constraints"), list)
                and isinstance(parsed_json.get("validation_plan"), list)
                and parsed_json.get("validation_plan")
            ) else 0.0
        if metric == "adversarial_criticism":
            return 1.0 if isinstance(parsed_json.get("flaws"), list) and parsed_json["flaws"] else 0.0
        return 0.0


class LocalModelRouter:
    def __init__(self, profile):
        self.profile = profile

    def select(self, model_class, task_class):
        candidates = self.candidates(model_class, task_class)
        return candidates[0] if candidates else None

    def candidates(self, model_class, task_class):
        if task_class and task_class not in CANONICAL_LLM_TASK_CLASSES:
            raise UnknownLLMTaskClass(
                f"UNKNOWN_LLM_TASK_CLASS requested_class={task_class} available_classes={sorted(CANONICAL_LLM_TASK_CLASSES)}"
            )
        configs = sorted(
            self.profile.get("configurations", []),
            key=lambda c: (
                tier_rank(c.get("model", {}).get("task_tier") or c.get("model", {}).get("profile")),
                c.get("ram_estimate", 0),
                -c.get("tokens_per_second", 0),
            ),
        )
        required_tasks = QUALITY_BY_CLASS.get(model_class, []) + ([task_class] if task_class else [])
        selected = []
        for config in configs:
            eligible = set(config.get("eligible_task_classes", []))
            if all(task in eligible for task in required_tasks if task in TASK_THRESHOLDS):
                selected.append(config)
        return semantic_configuration_candidates(selected)


class LocalInferenceManager:
    def __init__(self, root=None, hardware_probe=None, runtime_registry=None, model_registry=None):
        self.root = Path(root or os.environ.get("RESEARCH_LOCAL_PROFILE_ROOT", ".research-runs"))
        self.hardware_probe = hardware_probe or HardwareProbe()
        self.runtime_registry = runtime_registry or RuntimeRegistry()
        self.model_registry = model_registry or ModelRegistry()
        self.quant_policy = QuantizationPolicy()
        self.kv_policy = KVCachePolicy(os.environ.get("RESEARCH_LOCAL_KV_CACHE") or os.environ.get("RESEARCH_LOCAL_OPTIMIZATION", "balanced"))
        self.benchmark = CapabilityBenchmark()

    def build_profile(self):
        profile_path = self.root / "local_inference" / "profile.json"
        previous_profile = {}
        if profile_path.exists():
            try:
                previous_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_profile = {}
        hardware = self.hardware_probe.persist(self.root)
        runtimes = self.runtime_registry.discover()
        runtime_caps = {}
        for runtime in runtimes:
            if runtime.get("available") and runtime.get("backend_compatibility", "COMPATIBLE") == "COMPATIBLE":
                runtime_caps.update(runtime.get("capabilities", {}))
        configurations = []
        rejected_configurations = []
        max_context = int(os.environ.get("RESEARCH_LOCAL_MAX_CONTEXT", "2048"))
        for model in self.model_registry.candidates():
            runtime = self.runtime_registry.first_supporting(model)
            if not runtime:
                incompatible = self.runtime_registry.incompatibility_diagnostics()
                rejected_configurations.append({"model": model,
                    "reason": "backend_incompatible" if incompatible else "no_supporting_runtime",
                    "backend_diagnostics": incompatible})
                continue
            model_quant = QuantizationSpec(
                model.get("quantization", "UNKNOWN"),
                float(model.get("bits_per_weight", 8)),
                model.get("quantization", "unknown"),
                model.get("quality_hint", "unknown"),
            )
            for kv in self.kv_policy.candidates(runtime_caps):
                fit = estimate_fit(model, hardware, context=max_context, kv_cache=kv.__dict__)
                if not fit["fits_ram"] and not fit["fits_vram"]:
                    rejected_configurations.append({"model": model, "kv_quantization": kv.__dict__, "reason": "does_not_fit", "fit": fit})
                    continue
                config = {
                    "id": f"{runtime.name}:{model['id']}:{model_quant.id}:{kv.id}:{max_context}",
                    "runtime": runtime.name,
                    "model": model,
                    "quantization": model_quant.__dict__,
                    "kv_quantization": kv.__dict__,
                    "context": max_context,
                    "ram_estimate": fit["ram_estimate_bytes"],
                    "vram_estimate": fit["vram_estimate_bytes"],
                    "tokens_per_second": 0,
                }
                caps = self.benchmark.synthetic_scores(config)
                config["capability_prior"] = caps
                config["measured_capability"] = {}
                config["capabilities"] = {}
                config["eligible_task_classes"] = []
                configurations.append(config)
                self.benchmark.persist(self.root, config, config)
        # Discovery/smoke refreshes runtime facts, but must not erase benchmark
        # evidence.  Measured routing is a different lifecycle from discovery.
        previous_configs = {item.get("id"): item for item in previous_profile.get("configurations", [])}
        for config in configurations:
            measured = previous_configs.get(config.get("id"), {})
            if measured.get("measured_capability"):
                for key in ("measured_capability", "capabilities", "eligible_task_classes",
                            "tokens_per_second", "threads", "prompt_tokens_per_second",
                            "generation_tokens_per_second"):
                    if key in measured:
                        config[key] = measured[key]
        profile = {"hardware": hardware, "runtimes": runtimes, "configurations": configurations}
        if previous_profile.get("routing_profile"):
            profile["routing_profile"] = previous_profile["routing_profile"]
        if rejected_configurations:
            profile["rejected_configurations"] = rejected_configurations
        missing = self.model_registry.missing_from_env()
        if missing:
            profile["missing_models"] = missing
            profile.update(self.model_registry.missing_by_source())
        uninstalled = [
            {
                "id": spec.get("id"),
                "profile": spec.get("profile"),
                "source": spec.get("source"),
                "quantization": spec.get("quantization"),
                "reason": "configured_but_no_local_path",
            }
            for spec in self.model_registry.configured_model_specs()
            if not any(m.get("id") == spec.get("id") for m in self.model_registry.load().get("models", []))
        ]
        if uninstalled:
            profile["uninstalled_models"] = uninstalled
        path = profile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return profile

    def benchmark_configurations(self, thread_counts=None, max_configs=None):
        profile = self.build_profile()
        if os.environ.get("RESEARCH_LOCAL_SKIP_SMOKE_GATE", "0") != "1":
            smoke = self.smoke_test(profile=profile)
            if not smoke.get("pass"):
                result = {
                    **profile,
                    "benchmark_status": "REFUSED_SMOKE_FAILED",
                    "smoke": smoke,
                    "reason": "diagnostic smoke test could not obtain valid local model output",
                }
                path = self.root / "local_inference" / "profile.json"
                path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return result
        thread_counts = thread_counts or [int(x) for x in os.environ.get("RESEARCH_LOCAL_BENCH_THREADS", "2,4,6").split(",") if x]
        max_configs = max_configs or int(os.environ.get("RESEARCH_LOCAL_BENCH_MAX_CONFIGS", "9"))
        measured = []
        for config in profile.get("configurations", [])[:max_configs]:
            runtime = self.runtime_registry.first_supporting(config["model"])
            if not runtime:
                continue
            best = None
            for threads in thread_counts:
                result = self.benchmark.measured_benchmark(runtime, config, threads=threads)
                candidate = {**config, **result, "threads": threads}
                self.benchmark.persist(self.root, {**config, "id": f"{config['id']}:t{threads}"}, candidate)
                if best is None or len(candidate["eligible_task_classes"]) > len(best["eligible_task_classes"]):
                    best = candidate
            if best:
                measured.append(best)
        routing_profile = self._routing_profile(measured)
        output = {**profile, "configurations": measured, "routing_profile": routing_profile}
        profile_path = self.root / "local_inference" / "profile.json"
        profile_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        routing_path = self.root / "local_inference" / "routing_profile.json"
        routing_path.write_text(json.dumps(routing_profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    def smoke_test(self, thread_counts=None, max_configs=None, profile=None):
        profile = profile or self.build_profile()
        thread_counts = thread_counts or [int(os.environ.get("RESEARCH_LOCAL_THREADS", "2"))]
        max_configs = max_configs or 1
        results = []
        summary = {
            "status": "FAIL",
            "model": None,
            "quantization": None,
            "stdout": "",
            "stderr": "",
            "parsed_response": "",
            "prompt_tokens_per_second": None,
            "generation_tokens_per_second": None,
            "failure_class": "RUNTIME_ENVIRONMENT_FAILURE",
            "command": None,
        }
        for config in profile.get("configurations", [])[:max_configs]:
            runtime = self.runtime_registry.first_supporting(config["model"])
            if not runtime:
                continue
            result = self.benchmark.diagnostic_smoke(runtime, config, threads=thread_counts[0])
            output = {**config, "diagnostic_smoke": result}
            self.benchmark.persist(self.root, {**config, "id": f"{config['id']}:smoke"}, output)
            results.append(output)
            timing = result.get("timing_data", {})
            summary = {
                "status": "PASS" if result.get("status") == "SUCCESS" and result.get("parsed_json") == {"ok": True} else "FAIL",
                "model": config.get("model", {}).get("id"),
                "quantization": config.get("model", {}).get("quantization"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "parsed_response": result.get("parsed_assistant_output", ""),
                "prompt_tokens_per_second": timing.get("prompt_tokens_per_second"),
                "generation_tokens_per_second": timing.get("generation_tokens_per_second"),
                "failure_class": result.get("failure_class"),
                "command": result.get("command"),
                "backend_id": result.get("resource_diagnostics", {}).get("backend_id"),
                "invocation_mode": result.get("resource_diagnostics", {}).get("invocation_mode"),
                "executable": result.get("resource_diagnostics", {}).get("executable"),
                "structured_decoding": result.get("structured_decoding"),
                "reasoning_disabled": result.get("reasoning_disabled"),
                "jinja_disabled": result.get("jinja_disabled"),
                "json_schema_enabled": result.get("json_schema_enabled"),
                "schema_hash": result.get("schema_hash"),
                "extraction_strategy": result.get("extraction_strategy"),
                "parse_status": result.get("parse_status"),
            }
            if summary["status"] == "PASS":
                break
        output = {"pass": summary["status"] == "PASS", "summary": summary, "profile": profile, "smoke_tests": results}
        path = self.root / "local_inference" / "smoke.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    def _routing_profile(self, configs):
        tiers = {"FAST": None, "DEFAULT": None, "QUALITY": None}
        for config in configs:
            tier = str(config.get("model", {}).get("task_tier") or config.get("model", {}).get("profile") or "").upper()
            if tier in tiers and tiers[tier] is None:
                tiers[tier] = config["id"]
        assignments = {}
        for task in TASK_THRESHOLDS:
            eligible = [c for c in configs if task in c.get("eligible_task_classes", [])]
            assignments[task] = "HUMAN_REQUIRED" if not eligible else sorted(eligible, key=lambda c: tier_rank(c.get("model", {}).get("task_tier") or c.get("model", {}).get("profile")))[0]["id"]
        return {"tiers": tiers, "task_assignments": assignments, "paid_fallback": os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1"}

    def load_or_build_profile(self):
        path = self.root / "local_inference" / "profile.json"
        if path.exists():
            profile = json.loads(path.read_text(encoding="utf-8"))
            # Profiles written by older smoke implementations may have lost
            # their measured fields. Rehydrate only from immutable benchmark
            # artifacts; never synthesize eligibility.
            if not any(c.get("measured_capability") for c in profile.get("configurations", [])):
                benchmark_dir = self.root / "local_inference" / "benchmarks"
                measured_by_id = {}
                if benchmark_dir.exists():
                    for benchmark_path in benchmark_dir.glob("*.json"):
                        try:
                            measured = json.loads(benchmark_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            continue
                        if not measured.get("measured_capability"):
                            continue
                        config_id = measured.get("id")
                        current = measured_by_id.get(config_id)
                        if config_id and (current is None or len(measured.get("eligible_task_classes", [])) >
                                          len(current.get("eligible_task_classes", []))):
                            measured_by_id[config_id] = measured
                changed = False
                for config in profile.get("configurations", []):
                    measured = measured_by_id.get(config.get("id"))
                    if not measured:
                        continue
                    for key in ("measured_capability", "capabilities", "eligible_task_classes",
                                "tokens_per_second", "threads", "prompt_tokens_per_second",
                                "generation_tokens_per_second"):
                        if key in measured:
                            config[key] = measured[key]
                    changed = True
                routing_path = self.root / "local_inference" / "routing_profile.json"
                if routing_path.exists() and not profile.get("routing_profile"):
                    try:
                        profile["routing_profile"] = json.loads(routing_path.read_text(encoding="utf-8"))
                        changed = True
                    except (OSError, ValueError):
                        pass
                if changed:
                    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return profile
        return self.build_profile()

    def profile_metadata(self):
        path = self.root / "local_inference" / "profile.json"
        if not path.exists():
            return {"path": str(path), "sha256": None, "version": None}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"path": str(path), "sha256": sha256_file(path), "version": data.get("profile_version") or data.get("created_at")}

    def route(self, request, task_class=None):
        profile = self.load_or_build_profile()
        if not profile.get("routing_profile") and not any(c.get("measured_capability") for c in profile.get("configurations", [])):
            raise NoLocalProfileAvailable(f"NO_PROFILE_AVAILABLE path={self.root / 'local_inference' / 'profile.json'}")
        return LocalModelRouter(profile).select(request.requested_model_class, task_class)

    def route_candidates(self, request, task_class=None):
        profile = self.load_or_build_profile()
        if not profile.get("routing_profile") and not any(c.get("measured_capability") for c in profile.get("configurations", [])):
            raise NoLocalProfileAvailable(f"NO_PROFILE_AVAILABLE path={self.root / 'local_inference' / 'profile.json'}")
        return LocalModelRouter(profile).candidates(request.requested_model_class, task_class)


class LocalLLMProvider(LLMProvider):
    available = True

    def __init__(self, manager=None, fallback_provider=None):
        self.manager = manager or LocalInferenceManager()
        self.fallback_provider = fallback_provider
        self.context_reducer = ContextReducer()

    def generate(self, request):
        task_class = getattr(request, "task_class", None) or os.environ.get("RESEARCH_LOCAL_TASK_CLASS")
        try:
            config = self.manager.route(request, task_class)
        except NoLocalProfileAvailable:
            if self.fallback_provider and os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1":
                return self.fallback_provider.generate(request)
            raise
        if not config:
            if self.fallback_provider and os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1":
                return self.fallback_provider.generate(request)
            raise NoEligibleLocalModel(f"NO_ELIGIBLE_LOCAL_MODEL task_class={task_class or 'unspecified'}")
        prompt = self.context_reducer.reduce(request.prompt)
        runtime = self.manager.runtime_registry.first_supporting(config["model"])
        if runtime is None:
            incompatible = self.manager.runtime_registry.incompatibility_diagnostics()
            if incompatible:
                raise LocalInferenceBackendIncompatible(
                    "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE no socket-compatible local backend; "
                    + json.dumps(incompatible, sort_keys=True), [])
            raise NoEligibleLocalModel("NO_ELIGIBLE_LOCAL_MODEL no supporting local runtime backend")
        profile_meta = self.manager.profile_metadata()
        try:
            result = runtime.generate(prompt, {
                "id": config["id"],
                "model": config["model"],
                "context": config["context"],
                "kv_cache": config.get("kv_quantization", {}),
            })
        except Exception as exc:
            setattr(exc, "local_model", config["id"])
            raise
        return {
            "text": result["text"],
            "model": config["id"],
            "input_tokens": len(prompt.split()),
            "cached_input_tokens": 0,
            "output_tokens": len(result["text"].split()),
            "estimated_cost": 0.0,
            "actual_cost": 0.0,
            "local_configuration": config,
            "task_class": task_class,
            "selected_configuration_id": config["id"],
            "weight_quantization": config.get("model", {}).get("quantization"),
            "kv_quantization": config.get("kv_quantization", {}).get("id"),
            "local_profile": profile_meta,
            "status": "SUCCESS",
            "prompt_tokens_per_second": result.get("diagnostic", {}).get("timing_data", {}).get("prompt_tokens_per_second"),
            "generation_tokens_per_second": result.get("diagnostic", {}).get("timing_data", {}).get("generation_tokens_per_second"),
            "wall_clock_duration_seconds": result.get("duration_seconds"),
        }

    def generate_structured(self, request, schema=None, semantic_validator=None):
        task_class = getattr(request, "task_class", None) or os.environ.get("RESEARCH_LOCAL_TASK_CLASS")
        try:
            configs = self.manager.route_candidates(request, task_class)
        except NoLocalProfileAvailable:
            if self.fallback_provider and os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1":
                return self.fallback_provider.generate_structured(request, schema=schema)
            raise
        if not configs:
            if self.fallback_provider and os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1":
                return self.fallback_provider.generate_structured(request, schema=schema)
            raise NoEligibleLocalModel(f"NO_ELIGIBLE_LOCAL_MODEL task_class={task_class or 'unspecified'}")
        prompt = self.context_reducer.reduce(request.prompt)
        profile_meta = self.manager.profile_metadata()
        schema = schema or {"type": "object"}
        attempts = []
        infrastructure_retry_limit = max(0, int(os.environ.get("RESEARCH_LOCAL_INFRASTRUCTURE_RETRIES", "1")))

        def invoke(config, runtime, invocation_prompt, repair_attempt, semantic_retry_type=None):
            for infrastructure_retry in range(infrastructure_retry_limit + 1):
                result, attempt = self._structured_attempt(
                    runtime, config, invocation_prompt, schema, semantic_validator, task_class, profile_meta,
                    len(attempts) + 1, repair_attempt)
                attempt["infrastructure_retry_number"] = infrastructure_retry
                attempt["infrastructure_retry_budget"] = infrastructure_retry_limit
                attempt["retry_budget_type"] = "infrastructure" if infrastructure_retry else (
                    semantic_retry_type or ("structured_output" if repair_attempt else "initial"))
                attempts.append(attempt)
                if attempt.get("failure_type") not in {
                        "TRANSIENT_LOCAL_RUNTIME_FAILURE", "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"}:
                    return result, attempt
                if infrastructure_retry < infrastructure_retry_limit:
                    attempt["retry_reason"] = attempt["failure_type"]
                    time.sleep(min(float(os.environ.get("RESEARCH_LOCAL_INFRASTRUCTURE_RETRY_DELAY_SECONDS", "0.05")), 1.0))
            raise LocalRuntimeInfrastructureFailure(
                "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE bounded infrastructure retries exhausted", attempts)

        for config_index, config in enumerate(configs):
            runtime = self.manager.runtime_registry.first_supporting(config["model"])
            if runtime is None:
                incompatible = self.manager.runtime_registry.incompatibility_diagnostics()
                if incompatible:
                    raise LocalInferenceBackendIncompatible(
                        "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE no socket-compatible local backend; "
                        + json.dumps(incompatible, sort_keys=True), [])
                raise NoEligibleLocalModel("NO_ELIGIBLE_LOCAL_MODEL no supporting local runtime backend")
            result, attempt = invoke(
                config, runtime,
                prompt if config_index == 0 else structured_repair_prompt(prompt, schema, attempts), False,
                "model_quality_escalation" if config_index else None)
            if attempt["status"] == "SUCCESS":
                return structured_success_response(result, attempt, config, task_class, profile_meta, attempts)
            if attempt["failure_type"] == "STRUCTURED_DECODING_CONFIGURATION_FAILURE":
                raise StructuredDecodingConfigurationFailure("STRUCTURED_DECODING_CONFIGURATION_FAILURE local JSON-schema decoding failed", attempts)
            if config_index == 0:
                retry_prompt = prompt if attempt.get("failure_type") == "STRUCTURED_OUTPUT_INCOMPLETE" else structured_repair_prompt(prompt, schema, attempts)
                retry_type = "semantic" if attempt.get("failure_type") == "SEMANTIC_VALIDATION_FAILURE" else "structured_output"
                result, attempt = invoke(config, runtime, retry_prompt, True, retry_type)
                if attempt["status"] == "SUCCESS":
                    return structured_success_response(result, attempt, config, task_class, profile_meta, attempts)
                if attempt["failure_type"] == "STRUCTURED_DECODING_CONFIGURATION_FAILURE":
                    raise StructuredDecodingConfigurationFailure("STRUCTURED_DECODING_CONFIGURATION_FAILURE local JSON-schema decoding failed", attempts)
        last_failure_type = attempts[-1].get("failure_type") if attempts else None
        failure_type = last_failure_type or "MODEL_OUTPUT_INVALID"
        raise StructuredGenerationExhausted(f"{failure_type} local structured generation exhausted", attempts)

    def _structured_attempt(self, runtime, config, prompt, schema, semantic_validator, task_class, profile_meta, attempt_number, repair_attempt):
        started = time.time()
        result = {"text": "", "duration_seconds": None, "diagnostic": {}}
        data = None
        errors = []
        semantic_errors = []
        infrastructure_errors = []
        status = "SUCCESS"
        failure_type = None
        try:
            result = runtime.generate(prompt, structured_generation_config({
                "id": config["id"],
                "model": config["model"],
                "context": config["context"],
                "kv_cache": config.get("kv_quantization", {}),
            }, schema))
            data = parse_json_object(result.get("text", ""))
            if data is None:
                errors = ["response did not contain a JSON object"]
                status = "MODEL_OUTPUT_INVALID"
                failure_type = "MODEL_OUTPUT_INVALID"
            else:
                errors = validate_json_schema_subset(data, schema)
                if errors:
                    status = "SCHEMA_VALIDATION_FAILURE"
                    failure_type = "SCHEMA_VALIDATION_FAILURE"
                elif semantic_validator:
                    semantic_errors = list(semantic_validator(data) or [])
                    if semantic_errors:
                        status = "SEMANTIC_VALIDATION_FAILURE"
                        failure_type = "SEMANTIC_VALIDATION_FAILURE"
        except Exception as exc:
            diagnostic = getattr(exc, "diagnostic", {})
            result = {
                "text": diagnostic.get("parsed_assistant_output", ""),
                "duration_seconds": diagnostic.get("wall_clock_duration_seconds") or (time.time() - started),
                "diagnostic": diagnostic,
            }
            infrastructure_failure = classify_local_infrastructure_failure(diagnostic)
            if infrastructure_failure:
                infrastructure_errors = [str(exc)]
                errors = []
                status = infrastructure_failure
                failure_type = infrastructure_failure
            else:
                errors = [str(exc)]
                status = diagnostic.get("status") or "MODEL_EXECUTION_FAILED"
            if not infrastructure_failure and status in {
                "STRUCTURED_DECODING_CONFIGURATION_FAILURE",
                "ASSISTANT_OUTPUT_EXTRACTION_FAILURE",
                "STRUCTURED_OUTPUT_INCOMPLETE",
                "STRUCTURED_OUTPUT_INVALID_JSON",
                "EMPTY_OUTPUT",
            }:
                failure_type = status
            elif not infrastructure_failure:
                failure_type = "MODEL_EXECUTION_FAILED"
        diagnostic = result.get("diagnostic") or {}
        timing = diagnostic.get("timing_data", {})
        return result, {
            "stage": None,
            "task_class": task_class,
            "attempt_number": attempt_number,
            "repair_attempt": repair_attempt,
            "selected_configuration_id": config["id"],
            "actual_model": config["id"],
            "model_class": None,
            "weight_quantization": config.get("model", {}).get("quantization"),
            "kv_quantization": config.get("kv_quantization", {}).get("id"),
            "local_profile": profile_meta,
            "raw_response": result.get("text", ""),
            "command": diagnostic.get("command"),
            "stdout": diagnostic.get("stdout"),
            "stderr": diagnostic.get("stderr"),
            "exit_code": diagnostic.get("exit_code"),
            "assistant_generation_text": diagnostic.get("assistant_generation_text"),
            "isolated_assistant_text": diagnostic.get("isolated_assistant_text"),
            "extraction_strategy": diagnostic.get("extraction_strategy"),
            "extraction_diagnostics": diagnostic.get("extraction_diagnostics"),
            "schema": schema,
            "parsed_response": data,
            "schema_errors": errors,
            "semantic_errors": semantic_errors,
            "infrastructure_errors": infrastructure_errors,
            "subprocess_started": bool(diagnostic.get("command")),
            "generation_began": bool(diagnostic.get("assistant_generation_text")),
            "generation_produced": bool(result.get("text")),
            "resource_diagnostics": diagnostic.get("resource_diagnostics", {}),
            "status": status,
            "failure_type": failure_type,
            "input_tokens": len(prompt.split()),
            "cached_input_tokens": 0,
            "output_tokens": len((result.get("text") or "").split()),
            "estimated_cost": 0.0,
            "actual_cost": 0.0,
            "wall_clock_duration_seconds": result.get("duration_seconds") or (time.time() - started),
            "prompt_tokens_per_second": timing.get("prompt_tokens_per_second"),
            "generation_tokens_per_second": timing.get("generation_tokens_per_second"),
        }


def structured_success_response(result, attempt, config, task_class, profile_meta, attempts):
    return {
        "text": result["text"],
        "structured": attempt["parsed_response"],
        "model": config["id"],
        "input_tokens": attempt["input_tokens"],
        "cached_input_tokens": 0,
        "output_tokens": attempt["output_tokens"],
        "estimated_cost": 0.0,
        "actual_cost": 0.0,
        "local_configuration": config,
        "diagnostic": result.get("diagnostic"),
        "task_class": task_class,
        "selected_configuration_id": config["id"],
        "weight_quantization": config.get("model", {}).get("quantization"),
        "kv_quantization": config.get("kv_quantization", {}).get("id"),
        "local_profile": profile_meta,
        "status": "SUCCESS",
        "prompt_tokens_per_second": attempt.get("prompt_tokens_per_second"),
        "generation_tokens_per_second": attempt.get("generation_tokens_per_second"),
        "wall_clock_duration_seconds": attempt.get("wall_clock_duration_seconds"),
        "attempts": attempts,
    }


def compact_literature_item(item):
    if not isinstance(item, dict):
        return item
    return {
        key: (value[:240] if key.endswith("excerpt") and isinstance(value, str) else value)
        for key, value in item.items()
        if key in {"identifier", "title", "year", "abstract_excerpt", "verification_status"}
    }


def structured_repair_prompt(original_prompt, schema, attempts):
    last = attempts[-1] if attempts else {}
    validation_errors = (last.get("schema_errors", []) or []) + (last.get("semantic_errors", []) or [])
    if last.get("failure_type") == "SEMANTIC_VALIDATION_FAILURE" and len(schema.get("properties", {})) <= 3:
        reason = "The response did not address the requested substantive dimension. Give a fresh judgment; do not repeat these instructions."
        return "\n".join([
            str(original_prompt),
            "PREVIOUS OUTPUT WAS INVALID BECAUSE:",
            reason,
            "Answer the original question again. Return only the requested JSON object.",
        ])
    return json.dumps({
        "task": "Repair the previous response into one JSON object satisfying the schema. Do not add unrelated prose.",
        "required_schema": schema,
        "validation_errors": validation_errors,
        "previous_response": (last.get("raw_response") or "")[:4000],
        "original_task_summary": str(original_prompt)[:2000],
    }, sort_keys=True)


def semantic_configuration_candidates(configs):
    selected = []
    seen = set()
    for config in configs:
        model = config.get("model", {})
        key = (
            model.get("id"),
            model.get("quantization"),
            model.get("task_tier") or model.get("profile"),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(config)
    return selected


def contains_json_object(text):
    if not text:
        return False
    return parse_json_object(text) is not None


def parse_json_object(text):
    if not text:
        return None
    cleaned = strip_ansi(text).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[idx:])
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text or "")


def render_programmatic_llama_prompt(prompt):
    return " ".join(str(prompt or "").splitlines())


def extract_llama_assistant_output(stdout, stderr=None, prompt=None):
    return extract_llama_assistant_response(stdout, prompt).get("parsed_response", "")


def extract_llama_assistant_response(stdout, prompt=None, structured=False):
    stdout = strip_ansi(stdout or "")
    prompt = prompt or ""
    cleaned_stdout = remove_llama_diagnostics(stdout).strip()
    prompt_json_values = json_values_in_text(prompt)
    diagnostics = {
        "stdout_length": len(stdout),
        "cleaned_stdout_length": len(cleaned_stdout),
        "prompt_length": len(prompt),
        "prompt_json_count": len(prompt_json_values),
    }

    if not cleaned_stdout:
        return {
            "assistant_text": "",
            "parsed_response": "",
            "strategy": "empty_stdout",
            "status": "EMPTY_OUTPUT",
            "diagnostics": diagnostics,
        }

    span = isolate_llama_generation_span(stdout, prompt)
    diagnostics.update(span.get("diagnostics", {}))
    if span.get("status") != "SUCCESS":
        return {
            "assistant_text": span.get("assistant_text", ""),
            "parsed_response": "",
            "strategy": span.get("strategy"),
            "status": span.get("status"),
            "diagnostics": diagnostics,
        }

    segment = span["assistant_text"]
    if structured:
        parsed, parse_status, parse_diagnostics = parse_exact_top_level_json_text(segment)
        diagnostics.update(parse_diagnostics)
        return {
            "assistant_text": segment,
            "parsed_response": parsed,
            "strategy": span["strategy"],
            "status": parse_status,
            "diagnostics": diagnostics,
        }

    json_candidates = extract_json_texts(segment)
    diagnostics["assistant_json_candidates"] = len(json_candidates)
    if json_candidates:
        filtered = [
            candidate for candidate in json_candidates
            if parse_json_object(candidate) not in prompt_json_values
        ]
        diagnostics["assistant_non_prompt_json_candidates"] = len(filtered)
        if filtered:
            parsed = filtered[-1].strip()
            return {
                "assistant_text": segment,
                "parsed_response": parsed,
                "strategy": f"{span['strategy']}:last_non_prompt_json",
                "status": "SUCCESS",
                "diagnostics": diagnostics,
            }
    non_json = segment.strip()
    if non_json and not contains_only_prompt_json(non_json, prompt_json_values):
        return {
            "assistant_text": segment,
            "parsed_response": non_json,
            "strategy": f"{span['strategy']}:non_json_text",
            "status": "SUCCESS",
            "diagnostics": diagnostics,
        }

    return {
        "assistant_text": segment,
        "parsed_response": "",
        "strategy": f"{span['strategy']}:no_assistant_response",
        "status": "ASSISTANT_OUTPUT_EXTRACTION_FAILURE",
        "diagnostics": diagnostics,
    }


def isolate_llama_generation_span(stdout, prompt=None):
    text = strip_ansi(stdout or "")
    diagnostics = {}
    footer_patterns = (
        "\n[ Prompt:",
        "\n\n[ Prompt:",
        "\nExiting...",
        "\n\nExiting...",
        "\nllama_perf_context_print:",
        "\nllama_print_timings:",
    )

    def cut_footer(segment):
        cut = len(segment)
        for pattern in footer_patterns:
            idx = segment.find(pattern)
            if idx >= 0:
                cut = min(cut, idx)
        return remove_llama_diagnostics(segment[:cut]).strip()

    lines = text.splitlines()
    prompt_line_index = None
    for idx, line in enumerate(lines):
        if line.startswith("> "):
            prompt_line_index = idx
    if prompt_line_index is not None:
        segment = "\n".join(lines[prompt_line_index + 1:])
        segment = cut_footer(segment)
        diagnostics["prompt_display_line"] = prompt_line_index
        if not segment:
            return {
                "assistant_text": "",
                "strategy": "interactive_prompt_boundary_empty",
                "status": "ASSISTANT_OUTPUT_EXTRACTION_FAILURE",
                "diagnostics": diagnostics,
            }
        return {
            "assistant_text": segment,
            "strategy": "interactive_prompt_boundary",
            "status": "SUCCESS",
            "diagnostics": diagnostics,
        }

    bare = cut_footer(text)
    if bare.startswith("{") or bare.startswith("["):
        return {
            "assistant_text": bare,
            "strategy": "direct_stdout_json",
            "status": "SUCCESS",
            "diagnostics": diagnostics,
        }

    prompt = prompt or ""
    if prompt and prompt in text:
        segment = cut_footer(text.rsplit(prompt, 1)[1])
        if segment:
            return {
                "assistant_text": segment,
                "strategy": "exact_prompt_suffix",
                "status": "SUCCESS",
                "diagnostics": diagnostics,
            }
    return {
        "assistant_text": "",
        "strategy": "no_generation_boundary",
        "status": "ASSISTANT_OUTPUT_EXTRACTION_FAILURE",
        "diagnostics": diagnostics,
    }


def parse_exact_top_level_json_text(text):
    cleaned = (text or "").strip()
    diagnostics = {"assistant_generation_length": len(cleaned)}
    if not cleaned:
        return "", "EMPTY_OUTPUT", diagnostics
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        diagnostics["json_error"] = exc.msg
        diagnostics["json_error_pos"] = exc.pos
        status = "STRUCTURED_OUTPUT_INCOMPLETE" if looks_like_incomplete_json(cleaned) else "STRUCTURED_OUTPUT_INVALID_JSON"
        return "", status, diagnostics
    trailing = cleaned[end:].strip()
    diagnostics["json_end"] = end
    diagnostics["trailing_length"] = len(trailing)
    if not isinstance(value, dict):
        return "", "STRUCTURED_OUTPUT_INVALID_JSON", diagnostics
    if trailing:
        return "", "STRUCTURED_OUTPUT_INVALID_JSON", diagnostics
    return cleaned, "SUCCESS", diagnostics


def looks_like_incomplete_json(text):
    stack = []
    in_string = False
    escape = False
    pairs = {"}": "{", "]": "["}
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return in_string or bool(stack)


def extract_json_text(text):
    texts = extract_json_texts(text)
    return texts[0] if texts else ""


def extract_json_texts(text):
    cleaned = remove_llama_diagnostics(text)
    decoder = json.JSONDecoder()
    texts = []
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[idx:])
        except Exception:
            continue
        texts.append(cleaned[idx: idx + end].strip())
    return texts


def json_values_in_text(text):
    values = []
    for candidate in extract_json_texts(text or ""):
        value = parse_json_object(candidate)
        if value is not None:
            values.append(value)
    return values


def contains_only_prompt_json(text, prompt_json_values):
    values = json_values_in_text(text)
    return bool(values) and all(value in prompt_json_values for value in values)


def remove_llama_diagnostics(text):
    lines = []
    for line in (text or "").splitlines():
        lowered = line.strip().lower()
        if lowered.startswith(("llama_", "common_", "ggml_", "build:", "main:", "srv ", "slot ")):
            continue
        if "per token" in lowered and "tokens per second" in lowered:
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_llama_timing(text):
    data = {
        "prompt_tokens_per_second": None,
        "generation_tokens_per_second": None,
        "time_to_first_token_seconds": None,
        "raw_timing_output": "",
    }
    timing_lines = []
    for line in (text or "").splitlines():
        lowered = line.lower()
        if "tokens per second" in lowered or "tokens/s" in lowered or " t/s" in lowered or "prompt eval" in lowered or "eval time" in lowered:
            timing_lines.append(line)
            prompt_match = re.search(r"prompt eval.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", lowered)
            if prompt_match:
                data["prompt_tokens_per_second"] = float(prompt_match.group(1))
            gen_match = re.search(r"(?<!prompt )eval time.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", lowered)
            if gen_match:
                data["generation_tokens_per_second"] = float(gen_match.group(1))
            compact_match = re.search(r"prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\|\s*generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", lowered)
            if compact_match:
                data["prompt_tokens_per_second"] = float(compact_match.group(1))
                data["generation_tokens_per_second"] = float(compact_match.group(2))
    data["raw_timing_output"] = "\n".join(timing_lines)
    return data


def is_structured_decoding_error(stdout, stderr):
    text = f"{stdout or ''}\n{stderr or ''}".lower()
    return (
        "failed to initialize samplers" in text
        and ("error initializing grammar sampler" in text or "grammar" in text)
    )


def aggregate_timing(diagnostics):
    prompt = []
    generation = []
    for diagnostic in diagnostics:
        timing = diagnostic.get("timing_data", {})
        if timing.get("prompt_tokens_per_second") is not None:
            prompt.append(timing["prompt_tokens_per_second"])
        if timing.get("generation_tokens_per_second") is not None:
            generation.append(timing["generation_tokens_per_second"])
    return {
        "prompt_tokens_per_second": round(sum(prompt) / len(prompt), 3) if prompt else None,
        "generation_tokens_per_second": round(sum(generation) / len(generation), 3) if generation else None,
    }


def json_parse_mode(text, parsed_json):
    if parsed_json is None:
        return "NO_JSON"
    cleaned = strip_ansi(text or "").strip()
    try:
        loaded = json.loads(cleaned)
        if loaded == parsed_json:
            return "STRICT_JSON_ONLY"
    except Exception:
        pass
    return "JSON_FOUND"


def validate_json_schema_subset(data, schema):
    errors = []
    if not schema:
        return errors
    _validate_schema_value(data, schema, "$", errors)
    return errors


def _validate_schema_value(value, schema, path, errors):
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} expected const {schema['const']!r}")
        return
    expected_type = schema.get("type")
    if expected_type and not _json_type_matches(value, expected_type):
        errors.append(f"{path} expected type {expected_type}")
        return
    if isinstance(value, str):
        if "minLength" in schema and len(value.strip()) < int(schema["minLength"]):
            errors.append(f"{path} expected minLength {schema['minLength']}")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path} expected one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} expected >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} expected <= {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path} expected minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{idx}]", errors)
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"missing required key: {key}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value and isinstance(subschema, dict):
                _validate_schema_value(value[key], subschema, f"{path}.{key}", errors)
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            for key in sorted(extra):
                errors.append(f"{path} unexpected key: {key}")


def _json_type_matches(value, expected):
    expected_types = expected if isinstance(expected, list) else [expected]
    for item in expected_types:
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
        if item == "string" and isinstance(value, str):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "null" and value is None:
            return True
    return False


def classify_local_failure(diagnostic):
    status = diagnostic.get("status")
    stderr = (diagnostic.get("stderr") or "").lower()
    stdout = (diagnostic.get("stdout") or "").lower()
    exit_code = diagnostic.get("exit_code")
    if status in {"EXECUTION_FAILED", "TIMEOUT"} or (exit_code not in (None, 0)):
        return "RUNTIME_ENVIRONMENT_FAILURE"
    if "failed to get a free port" in stderr or "failed to get a free port" in stdout:
        return "RUNTIME_ENVIRONMENT_FAILURE"
    if status in {"EMPTY_OUTPUT", "PARSE_FAILED"}:
        return "MODEL_OUTPUT_FAILURE"
    return None


def classify_local_infrastructure_failure(diagnostic):
    if diagnostic.get("status") == "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE":
        return "LOCAL_INFERENCE_BACKEND_INCOMPATIBLE"
    stderr = (diagnostic.get("stderr") or "").lower()
    stdout = (diagnostic.get("stdout") or "").lower()
    generated = (diagnostic.get("assistant_generation_text") or diagnostic.get("parsed_assistant_output") or "").strip()
    if generated:
        return None
    combined = stderr + "\n" + stdout
    if "failed to get a free port" in combined:
        return "TRANSIENT_LOCAL_RUNTIME_FAILURE"
    if any(marker in combined for marker in (
            "failed to listen on socket", "address already in use", "too many open files",
            "resource temporarily unavailable")):
        return "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"
    return None


def tier_rank(tier):
    return {"FAST": 0, "DEFAULT": 1, "QUALITY": 2}.get(str(tier or "").upper(), 99)
