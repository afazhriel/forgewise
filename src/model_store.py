"""Portable model acquisition for ForgeWise.

The trained model binary (``models/model.joblib``, ~15 MB) is intentionally
**not** committed to Git. On a fresh deployment (e.g. Streamlit Community Cloud)
this module:

1. reuses ``models/model.joblib`` if it is present (local development); or
2. trains the deterministic baseline from the official UCI AI4I dataset, which
   ``src.data_loader`` downloads automatically, and caches the result.

Training is cached in-process (``lru_cache``) so it happens at most once per
process, and the Streamlit UI additionally wraps this in ``st.cache_resource``.

Any failure is surfaced as a clear :class:`ModelUnavailable` so the UI can show
an actionable message instead of crashing. No executable artifacts are fetched
from untrusted sources: the only download is the official UCI dataset over
HTTPS, whose schema is validated before training.
"""

from __future__ import annotations

from functools import lru_cache

from src import data_loader, train_model


class ModelUnavailable(RuntimeError):
    """Raised when no model can be loaded or trained."""


def model_artifact_exists() -> bool:
    """True when both the model binary and its metadata are present on disk."""
    return train_model.MODEL_PATH.exists() and train_model.METADATA_PATH.exists()


def _bootstrap(verbose: bool) -> tuple[object, dict]:
    if model_artifact_exists():
        try:
            return train_model.load_model()
        except Exception as exc:  # corrupt / incompatible artifact
            raise ModelUnavailable(
                f"Found a model artifact at {train_model.MODEL_PATH} but could "
                f"not load it: {exc}"
            ) from exc

    try:
        return train_model.train(verbose=verbose)
    except Exception as exc:  # network / dataset / training failure
        raise ModelUnavailable(
            "No trained model artifact was found and the model could not be "
            "trained at startup. Training requires the official UCI AI4I 2020 "
            f"dataset, downloaded automatically from {data_loader.SOURCE_URL}. "
            f"Underlying error: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_model() -> tuple[object, dict]:
    """Return ``(pipeline, metadata)``, loading or training as needed (cached)."""
    return _bootstrap(verbose=False)


def clear_cache() -> None:
    """Clear the in-process model cache (used by tests)."""
    get_model.cache_clear()
