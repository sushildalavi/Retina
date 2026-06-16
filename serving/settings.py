from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field

from serving.errors import RetinaConfigurationError, RetinaMissingArtifactError


class RetinaSettings(BaseModel):
    config_path: Path
    project_name: str = "Retina"
    dataset_name: str = "flickr8k"
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    metadata_path: Path
    index_path: Path
    index_meta_path: Path
    reports_dir: Path
    image_artifact_dir: Path = Path("data/artifacts/images")
    setup_command: str = "make all-local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RetinaSettings":
        config_path = Path(path)
        if not config_path.exists():
            raise RetinaConfigurationError(
                f"Config file not found: {config_path}",
                details={"setup_command": "make all-local", "config_path": str(config_path)},
            )
        payload = yaml.safe_load(config_path.read_text()) or {}
        artifacts: Dict[str, Any] = dict(payload.get("artifacts", {}) or {})
        server: Dict[str, Any] = dict(payload.get("server", {}) or {})
        return cls(
            config_path=config_path,
            project_name=str(payload.get("project", {}).get("name", "Retina")),
            dataset_name=str(payload.get("dataset", {}).get("name", payload.get("dataset_name", "flickr8k"))),
            model_name=str(payload.get("model", {}).get("name", "openai/clip-vit-base-patch32")),
            device=str(payload.get("model", {}).get("device", "auto")),
            cors_origins=list(server.get("cors_origins", ["http://localhost:5173"])),
            metadata_path=Path(artifacts.get("metadata_path", "data/processed/retina_metadata.jsonl")),
            index_path=Path(artifacts.get("index_path", "data/processed/index/retina_image_index.faiss")),
            index_meta_path=Path(artifacts.get("index_meta_path", "data/processed/index/retina_image_index.json")),
            reports_dir=Path(artifacts.get("reports_dir", "reports")),
            image_artifact_dir=Path(artifacts.get("image_dir", "data/artifacts/images")),
            setup_command=str(payload.get("setup", {}).get("command", "make all-local")),
            api_host=str(server.get("host", "127.0.0.1")),
            api_port=int(server.get("port", 8000)),
        )

    def check_artifacts(self) -> None:
        missing = []
        for label, path in (
            ("metadata_path", self.metadata_path),
            ("index_path", self.index_path),
            ("index_meta_path", self.index_meta_path),
            ("image_artifact_dir", self.image_artifact_dir),
            ("reports_dir", self.reports_dir),
        ):
            if not Path(path).exists():
                missing.append({"artifact": label, "path": str(path)})
        if missing:
            raise RetinaMissingArtifactError(
                "Required Retina artifacts are missing",
                details={"missing": missing, "setup_command": self.setup_command},
            )

    def report_paths(self) -> Dict[str, str]:
        names = {
            "final_summary": "final_retina_metrics_summary.json",
            "retrieval_eval": "flickr8k_full_retrieval_eval.json",
            "recommendation_eval": "flickr8k_full_recommendation_eval.json",
            "random_baseline": "flickr8k_full_random_baseline.json",
            "runtime_benchmark": "flickr8k_full_runtime_benchmark.json",
            "failure_analysis": "flickr8k_full_retrieval_failures.md",
            "model_baseline": "model_baseline_comparison.json",
            "scaling_experiment": "scaling_experiment.json",
        }
        return {name: str(self.reports_dir / filename) for name, filename in names.items()}
