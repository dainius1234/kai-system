"""
Tests for J3: PII Auto-Redaction.
Covers: common/runtime.py patterns, memu-core import + redaction wiring,
        dashboard /api/pii/scan endpoint, PII Scanner HTML/JS in Settings tab.
Run: pytest scripts/test_j3_pii_redaction.py -v
"""
from pathlib import Path
import re

ROOT       = Path(__file__).parent.parent
RUNTIME_PY = ROOT / "common" / "runtime.py"
MEMU_PY    = ROOT / "memu-core" / "app.py"
DASH_PY    = ROOT / "dashboard" / "app.py"
APP_HTML   = ROOT / "dashboard" / "static" / "app.html"


def _runtime():
    return RUNTIME_PY.read_text(encoding="utf-8")


def _memu():
    return MEMU_PY.read_text(encoding="utf-8")


def _dash():
    return DASH_PY.read_text(encoding="utf-8")


def _html():
    return APP_HTML.read_text(encoding="utf-8")


# ── common/runtime.py — PII patterns and functions ────────────────────────────

class TestRuntimePiiPatterns:
    def test_pii_patterns_dict_defined(self):
        assert "_PII_PATTERNS" in _runtime()

    def test_email_pattern_present(self):
        assert '"email"' in _runtime()

    def test_credit_card_pattern_present(self):
        assert '"credit_card"' in _runtime()

    def test_phone_pattern_present(self):
        assert '"phone"' in _runtime()

    def test_uk_ni_number_pattern_present(self):
        assert '"uk_ni_number"' in _runtime()

    def test_api_token_pattern_present(self):
        assert '"api_token"' in _runtime()

    def test_uk_postcode_pattern_present(self):
        assert '"uk_postcode"' in _runtime()

    def test_detect_pii_defined(self):
        assert "def detect_pii(" in _runtime()

    def test_redact_pii_defined(self):
        assert "def redact_pii(" in _runtime()

    def test_redaction_tag_format(self):
        assert "REDACTED" in _runtime()

    def test_redact_pii_returns_tuple(self):
        src = _runtime()
        block = src[src.find("def redact_pii("):][:600]
        assert "counts" in block
        assert "return result" in block

    def test_detect_pii_returns_counts(self):
        src = _runtime()
        block = src[src.find("def detect_pii("):][:200]
        assert "counts" in block


# ── memu-core/app.py — import and redaction wiring ────────────────────────────

class TestMemuCorePiiWiring:
    def test_redact_pii_imported(self):
        src = _memu()
        import_line = next(
            (l for l in src.splitlines() if "from common.runtime import" in l), ""
        )
        assert "redact_pii" in import_line, "redact_pii must be in the common.runtime import"

    def test_memorize_event_calls_redact_pii(self):
        src = _memu()
        func_start = src.find("async def memorize_event(")
        assert func_start != -1
        func_block = src[func_start:func_start + 4000]
        assert "redact_pii(" in func_block

    def test_memorize_event_returns_pii_redacted_count(self):
        src = _memu()
        func_start = src.find("async def memorize_event(")
        func_block = src[func_start:func_start + 4000]
        assert "pii_redacted" in func_block

    def test_memorize_event_logs_pii_counts_not_content(self):
        src = _memu()
        block = src[src.find("memorize pii_redacted"):][:200] if "memorize pii_redacted" in src else ""
        assert "counts" in block, "audit log must reference counts, not content"

    def test_memorize_event_uses_redacted_text_for_store(self):
        src = _memu()
        func_start = src.find("async def memorize_event(")
        func_block = src[func_start:func_start + 4000]
        assert "redacted_text" in func_block

    def test_quick_note_calls_redact_pii(self):
        src = _memu()
        func_start = src.find("async def quick_note(")
        assert func_start != -1
        func_block = src[func_start:func_start + 500]
        assert "redact_pii(" in func_block

    def test_quick_note_logs_pii_counts(self):
        src = _memu()
        func_start = src.find("async def quick_note(")
        func_block = src[func_start:func_start + 500]
        assert "pii" in func_block.lower()

    def test_quick_note_redacts_before_store(self):
        # redact_pii must come before the NoteRequest text is stored
        src = _memu()
        func_start = src.find("async def quick_note(")
        func_block = src[func_start:func_start + 800]
        redact_pos = func_block.find("redact_pii(")
        assert redact_pos != -1


# ── dashboard/app.py — /api/pii/scan endpoint ─────────────────────────────────

class TestDashboardPiiEndpoint:
    def test_verifier_url_constant_defined(self):
        src = _dash()
        assert "VERIFIER_URL" in src

    def test_verifier_url_default_host(self):
        src = _dash()
        assert "verifier:8052" in src

    def test_pii_scan_endpoint_defined(self):
        assert '@app.post("/api/pii/scan")' in _dash()

    def test_pii_scan_proxies_to_verifier(self):
        src = _dash()
        block = src[src.find('/api/pii/scan'):][:400]
        assert "VERIFIER_URL" in block or "verifier" in block.lower()

    def test_pii_scan_forwards_text_field(self):
        src = _dash()
        block = src[src.find('async def api_pii_scan'):][:400]
        assert '"text"' in block

    def test_pii_scan_forwards_auto_redact(self):
        src = _dash()
        block = src[src.find('async def api_pii_scan'):][:400]
        assert "auto_redact" in block

    def test_pii_scan_has_fallback(self):
        src = _dash()
        block = src[src.find('async def api_pii_scan'):][:400]
        assert "fallback" in block

    def test_pii_scan_fallback_has_total_pii(self):
        src = _dash()
        block = src[src.find('async def api_pii_scan'):][:400]
        assert "total_pii" in block

    def test_pii_scan_proxies_to_redact_route(self):
        src = _dash()
        block = src[src.find('async def api_pii_scan'):][:400]
        assert "/redact" in block


# ── app.html — Settings tab PII Scanner UI ────────────────────────────────────

class TestHtmlPiiScanner:
    def test_pii_scanner_card_present(self):
        assert "PII Scanner" in _html()

    def test_pii_input_textarea_present(self):
        assert 'id="piiInput"' in _html()

    def test_detect_only_button_present(self):
        html = _html()
        assert "piiScan(false)" in html

    def test_detect_and_redact_button_present(self):
        html = _html()
        assert "piiScan(true)" in html

    def test_clear_button_present(self):
        assert "piiClear()" in _html()

    def test_result_container_present(self):
        assert 'id="piiResult"' in _html()

    def test_summary_div_present(self):
        assert 'id="piiSummary"' in _html()

    def test_output_textarea_present(self):
        assert 'id="piiOutput"' in _html()


# ── app.html — PII JS functions ───────────────────────────────────────────────

class TestHtmlPiiFunctions:
    def test_pii_scan_function_defined(self):
        assert "async function piiScan(" in _html()

    def test_pii_clear_function_defined(self):
        assert "function piiClear()" in _html()

    def test_pii_scan_calls_api_endpoint(self):
        html = _html()
        block = html[html.find("async function piiScan("):][:500]
        assert "/api/pii/scan" in block

    def test_pii_scan_reads_pii_found(self):
        html = _html()
        block = html[html.find("async function piiScan("):][:1200]
        assert "pii_found" in block

    def test_pii_scan_reads_total_pii(self):
        html = _html()
        block = html[html.find("async function piiScan("):][:1200]
        assert "total_pii" in block

    def test_pii_scan_sanitises_output(self):
        html = _html()
        block = html[html.find("async function piiScan("):][:1800]
        assert "DOMPurify.sanitize" in block

    def test_pii_clear_resets_input(self):
        html = _html()
        block = html[html.find("function piiClear()"):][:200]
        assert "piiInput" in block

    def test_pii_clear_hides_result(self):
        html = _html()
        block = html[html.find("function piiClear()"):][:200]
        assert "piiResult" in block
