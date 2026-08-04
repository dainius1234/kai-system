"""Global test configuration — runs before any test imports."""

import os
import sys

# Allow dev HMAC secret in test environment to avoid RuntimeError
# in common.auth._secret() when INTERSERVICE_HMAC_SECRET is not set.
os.environ.setdefault("HMAC_ALLOW_DEV_SECRET", "true")


# agentic/app.py writes the soul document to
# `Path(os.getenv("SOUL_PATH", "/data/SOUL.md"))`, and falls back to the
# *relative* `data/SOUL.md` — which, with pytest's cwd at the repo root, is
# the repository's own copy. A test that exercises that endpoint therefore
# rewrites a tracked file, and the tests that assert on its contents then
# depend on whether they ran before or after it.
#
# Invisible here only because `git checkout -- data/` had become a reflex
# between local runs. CI has no such reflex: on its first complete run
# `test_j_series` failed on "Core Values" missing and `test_soul_identity`
# on the file being under 100 bytes — both reading a document another test
# had replaced.
import tempfile as _tempfile

_soul_home = _tempfile.mkdtemp(prefix="kai-soul-")
os.environ.setdefault("SOUL_PATH", os.path.join(_soul_home, "SOUL.md"))


# ── The ML stack is blocked during tests ─────────────────────────────
# `memu-core/app.py` loads an embedding model at *module* scope:
#
#     from sentence_transformers import SentenceTransformer as _ST
#     _st_model = _ST(EMBEDDING_MODEL_NAME)
#
# On a machine without sentence-transformers that raises ImportError and
# the fallback runs — which is what MEMU_ALLOW_FAKE_EMBEDDINGS exists for.
# On a machine *with* it, GitHub's runner, the import succeeds and drags in
# transformers, then torch, then the CUDA bindings, on a box with no GPU.
# On 2026-08-04 that took pytest down with **SIGSEGV (exit 139)** during
# collection: not a failed test, a dead process — so no summary line, no
# isolation report, and nothing naming the file responsible.
#
# Blocking the import makes CI behave exactly as every developer machine
# already does, which is the environment the suite is green in. That this
# is safe is not an opinion: these packages are absent locally and 4,220
# tests pass, so nothing in the suite can depend on them.
#
# A *test* decision, not a production one. That memu-core cannot be
# imported without loading a model stays true and stays visible; it is
# simply not something a unit test should pay for. Set
# KAI_TESTS_USE_REAL_ML=true to lift the block.
# The same argument extends past the ML stack. A dependency that the tests
# *stub* and that no developer machine has is one where the suite has only
# ever exercised the stubbed branch. If CI has it installed, CI silently
# runs different code — which is how both the OpenCV crash and the segfault
# happened, an hour apart, from opposite ends of the dependency list.
#
# So: everything below is stubbed somewhere in scripts/test_*.py AND absent
# on developer machines. Blocking it cannot change the local result (it is
# already unimportable here) and makes CI's result identical. Verified, not
# assumed — the full suite is run with a landmine module for every name
# here, each raising SystemExit if imported.
#
# `psutil`, `redis`, `docx` and `feedparser` are deliberately NOT here.
# They are installed locally, so the suite genuinely exercises them, and
# blocking them would change the answer rather than fix a divergence.
_BLOCKED = {
    # ML stack — memu-core loads a model at import; CI segfaulted on it.
    "sentence_transformers", "transformers", "torch", "deepface",
    # OpenCV — CI's build imports but has no CascadeClassifier.
    "cv2",
    # Datastores and clients the tests always fake.
    "psycopg2", "lakefs_client", "aioredis",
    # Service-specific optional dependencies.
    "caldav", "icalendar", "docker", "watchdog", "letta", "yfinance",
    # document-parser's optional format readers.
    "fitz", "ezdxf", "xlrd", "pptx", "openpyxl", "bs4",
}


class _BlockHeavyML:
    """A meta-path finder that refuses the ML stack, as if uninstalled.

    Raising ModuleNotFoundError rather than substituting a mock is
    deliberate: every call site already has an `except ImportError`
    fallback written for exactly this case and exercised constantly. A
    mock would take a *different* branch — one no developer machine runs —
    and two environments disagreeing is the whole problem here.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(
                f"{fullname} is blocked during tests (conftest.py). "
                "Set KAI_TESTS_USE_REAL_ML=true to allow it."
            )
        return None


if os.getenv("KAI_TESTS_USE_REAL_ML", "false").lower() not in {"1", "true", "yes"}:
    # Only block what has not already been imported: pretending otherwise
    # would be a third behaviour rather than a second.
    _BLOCKED -= {name for name in _BLOCKED if name in sys.modules}
    if _BLOCKED:
        sys.meta_path.insert(0, _BlockHeavyML())
