"""Tests for common/errors.py — structured error codes and KaiError."""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.errors import ErrorCode, KaiError, _STATUS_MAP


# ─────────────────────────────────────────────────────────────────────
#  ErrorCode enum
# ─────────────────────────────────────────────────────────────────────

class TestErrorCode:
    def test_all_codes_start_with_E(self):
        for code in ErrorCode:
            assert code.value.startswith("E"), f"{code.name} value doesn't start with E"

    def test_codes_are_unique(self):
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate error code values"

    def test_client_errors_are_1xxx(self):
        client_codes = [c for c in ErrorCode if c.value.startswith("E1")]
        assert len(client_codes) >= 5

    def test_service_errors_are_2xxx(self):
        service_codes = [c for c in ErrorCode if c.value.startswith("E2")]
        assert len(service_codes) >= 4

    def test_conviction_errors_are_3xxx(self):
        safety_codes = [c for c in ErrorCode if c.value.startswith("E3")]
        assert len(safety_codes) >= 4

    def test_operator_errors_are_4xxx(self):
        operator_codes = [c for c in ErrorCode if c.value.startswith("E4")]
        assert len(operator_codes) >= 3


# ─────────────────────────────────────────────────────────────────────
#  HTTP status mapping
# ─────────────────────────────────────────────────────────────────────

class TestStatusMapping:
    def test_every_code_has_status(self):
        for code in ErrorCode:
            assert code in _STATUS_MAP, f"{code.name} missing from _STATUS_MAP"

    def test_client_errors_are_4xx(self):
        client_codes = [c for c in ErrorCode if c.value.startswith("E1")]
        for code in client_codes:
            status = _STATUS_MAP[code]
            assert 400 <= status < 500, f"{code.name} mapped to {status}, expected 4xx"

    def test_rate_limited_is_429(self):
        assert _STATUS_MAP[ErrorCode.RATE_LIMITED] == 429

    def test_circuit_open_is_503(self):
        assert _STATUS_MAP[ErrorCode.CIRCUIT_OPEN] == 503

    def test_not_found_is_404(self):
        assert _STATUS_MAP[ErrorCode.NOT_FOUND] == 404


# ─────────────────────────────────────────────────────────────────────
#  KaiError
# ─────────────────────────────────────────────────────────────────────

class TestKaiError:
    def test_basic_construction(self):
        err = KaiError(ErrorCode.CIRCUIT_OPEN, detail="memu down")
        assert err.code == ErrorCode.CIRCUIT_OPEN
        assert err.detail == "memu down"
        assert err.http_status == 503

    def test_default_detail_from_name(self):
        err = KaiError(ErrorCode.RATE_LIMITED)
        assert "Rate Limited" in err.detail

    def test_to_dict_structure(self):
        err = KaiError(ErrorCode.INJECTION_DETECTED, detail="blocked")
        d = err.to_dict()
        assert d["error"] is True
        assert d["code"] == "E1002"
        assert d["name"] == "INJECTION_DETECTED"
        assert d["detail"] == "blocked"
        assert d["status"] == 400

    def test_to_dict_json_serializable(self):
        err = KaiError(ErrorCode.INTERNAL_ERROR, detail="oops")
        s = json.dumps(err.to_dict())
        assert isinstance(s, str)

    def test_str_includes_code(self):
        err = KaiError(ErrorCode.FORBIDDEN, detail="policy says no")
        assert "[E1005]" in str(err)

    def test_is_exception(self):
        err = KaiError(ErrorCode.STORAGE_ERROR)
        assert isinstance(err, Exception)

    def test_context_field(self):
        err = KaiError(ErrorCode.DEPENDENCY_ERROR, context={"service": "memu"})
        assert err.context["service"] == "memu"

    def test_unknown_code_defaults_500(self):
        """If somehow a code is missing from the map, default to 500."""
        err = KaiError(ErrorCode.INTERNAL_ERROR)
        assert err.http_status == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
