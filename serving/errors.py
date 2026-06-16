from __future__ import annotations

from typing import Any, Dict


class RetinaServiceError(Exception):
    code = "retina_error"
    status_code = 500

    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RetinaConfigurationError(RetinaServiceError):
    code = "configuration_error"
    status_code = 500


class RetinaMissingArtifactError(RetinaServiceError):
    code = "missing_artifacts"
    status_code = 503


class RetinaModelLoadError(RetinaServiceError):
    code = "model_load_error"
    status_code = 503


class RetinaNotFoundError(RetinaServiceError):
    code = "not_found"
    status_code = 404


class RetinaInvalidRequestError(RetinaServiceError):
    code = "invalid_request"
    status_code = 422
