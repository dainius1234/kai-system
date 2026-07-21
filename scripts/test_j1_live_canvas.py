"""
Tests for J1: Live Canvas D3 visualization.
Verifies static assets, HTML structure, JS function presence, and data endpoints.
Run: pytest scripts/test_j1_live_canvas.py -v
"""
from pathlib import Path
import re

ROOT = Path(__file__).parent.parent
APP_HTML = ROOT / "dashboard" / "static" / "app.html"
STATIC   = ROOT / "dashboard" / "static"
DASH_PY  = ROOT / "dashboard" / "app.py"


def _html():
    return APP_HTML.read_text(encoding="utf-8")


# ── Static Assets ──────────────────────────────────────────────────────────────

class TestStaticAssets:
    def test_d3_file_exists(self):
        assert (STATIC / "d3.v7.min.js").exists(), "d3.v7.min.js must exist in dashboard/static/"

    def test_d3_file_non_trivial(self):
        size = (STATIC / "d3.v7.min.js").stat().st_size
        assert size > 100_000, f"d3.v7.min.js looks too small ({size} bytes) — may be truncated"

    def test_d3_script_tag_in_html(self):
        html = _html()
        assert "d3.v7.min.js" in html, "HTML must load D3 via /static/d3.v7.min.js script tag"

    def test_d3_loaded_before_body(self):
        html = _html()
        script_pos = html.find("d3.v7.min.js")
        body_pos   = html.find("<body")
        assert script_pos < body_pos, "D3 script tag must be in <head>, before <body>"


# ── HTML Structure ─────────────────────────────────────────────────────────────

class TestHtmlStructure:
    def test_canvas_view_section_exists(self):
        assert 'id="canvasView"' in _html()

    def test_canvas_d3_container(self):
        assert 'id="canvasD3"' in _html(), "canvasD3 div must exist (D3 SVG target)"

    def test_no_plain_canvas_element(self):
        html = _html()
        # The old <canvas id="liveCanvas"> should be gone
        assert 'id="liveCanvas"' not in html, "Old plain canvas element should be removed"

    def test_mode_buttons_present(self):
        html = _html()
        assert 'id="btnMindmap"'  in html
        assert 'id="btnEmotion"'  in html
        assert 'id="btnPlan"'     in html

    def test_canvas_nav_item(self):
        assert 'data-view="canvas"' in _html()

    def test_canvas_legend_span(self):
        assert 'id="canvasLegend"' in _html()

    def test_canvas_status_div(self):
        assert 'id="canvasStatus"' in _html()


# ── JavaScript Functions ───────────────────────────────────────────────────────

class TestJavaScriptFunctions:
    def test_refresh_canvas_defined(self):
        assert "async function refreshCanvas()" in _html()

    def test_canvas_mode_defined(self):
        assert "function canvasMode(" in _html()

    def test_draw_canvas_defined(self):
        assert "function drawCanvas(" in _html()

    def test_draw_mind_map_d3(self):
        assert "_drawMindMap(" in _html()

    def test_draw_emotion_timeline_d3(self):
        assert "_drawEmotionTimeline(" in _html()

    def test_draw_plan_flow_d3(self):
        assert "_drawPlanFlow(" in _html()

    def test_force_simulation_used(self):
        assert "d3.forceSimulation" in _html()

    def test_d3_zoom_used(self):
        assert "d3.zoom()" in _html()

    def test_d3_drag_used(self):
        assert "d3.drag()" in _html()

    def test_emotion_timeline_endpoint_fetched(self):
        assert "/api/emotion/timeline" in _html()

    def test_thinking_endpoint_fetched(self):
        assert "/api/thinking" in _html()

    def test_switch_view_triggers_refresh_canvas(self):
        html = _html()
        # switchView('canvas') should call refreshCanvas()
        canvas_block = html[html.find("if (name === 'canvas')"):][:100]
        assert "refreshCanvas()" in canvas_block

    def test_canvas_mode_button_toggle(self):
        # canvasMode() should update button backgrounds
        html = _html()
        assert "var(--accent)" in html[html.find("function canvasMode("):][:300]


# ── Dashboard API Endpoints ────────────────────────────────────────────────────

class TestDashboardEndpoints:
    def test_goals_endpoint_exists(self):
        assert '@app.get("/api/goals")' in DASH_PY.read_text()

    def test_emotion_timeline_endpoint_exists(self):
        assert '/api/emotion/timeline' in DASH_PY.read_text()

    def test_thinking_endpoint_exists(self):
        assert '@app.get("/api/thinking")' in DASH_PY.read_text()

    def test_memory_stats_endpoint_exists(self):
        assert '/api/memory/stats' in DASH_PY.read_text()


# ── D3 API Usage Patterns ──────────────────────────────────────────────────────

class TestD3Patterns:
    def test_svg_appended(self):
        assert ".append('svg')" in _html()

    def test_force_link_used(self):
        assert "d3.forceLink" in _html()

    def test_curve_monotone_used(self):
        assert "d3.curveMonotoneX" in _html()

    def test_scale_time_used(self):
        assert "d3.scaleTime()" in _html()

    def test_scale_linear_used(self):
        assert "d3.scaleLinear()" in _html()

    def test_axis_bottom_used(self):
        assert "d3.axisBottom(" in _html()

    def test_area_chart_used(self):
        assert "d3.area()" in _html()

    def test_line_chart_used(self):
        assert "d3.line()" in _html()
