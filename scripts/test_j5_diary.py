"""
Tests for J5: Memory Diary / Viewer.
Covers: new /api/memories/recent endpoint, HTML structure, JS functions, filter controls.
Run: pytest scripts/test_j5_diary.py -v
"""
from pathlib import Path

ROOT     = Path(__file__).parent.parent
APP_HTML = ROOT / "dashboard" / "static" / "app.html"
DASH_PY  = ROOT / "dashboard" / "app.py"


def _html():
    return APP_HTML.read_text(encoding="utf-8")


def _dash():
    return DASH_PY.read_text(encoding="utf-8")


# ── Backend Endpoint ───────────────────────────────────────────────────────────

class TestRecentEndpoint:
    def test_recent_endpoint_defined(self):
        assert '@app.get("/api/memories/recent")' in _dash()

    def test_recent_uses_memory_retrieve(self):
        src = _dash()
        block = src[src.find('/api/memories/recent'):][:400]
        assert "/memory/retrieve" in block

    def test_recent_wraps_as_records(self):
        src = _dash()
        block = src[src.find('/api/memories/recent'):][:600]
        assert '"records"' in block

    def test_recent_has_top_k_param(self):
        src = _dash()
        block = src[src.find('/api/memories/recent'):][:300]
        assert "top_k" in block

    def test_existing_memories_endpoint_unchanged(self):
        src = _dash()
        assert '@app.get("/api/memories")' in src


# ── HTML Structure ─────────────────────────────────────────────────────────────

class TestHtmlStructure:
    def test_diary_view_section(self):
        assert 'id="diaryView"' in _html()

    def test_diary_cards_container(self):
        assert 'id="diaryCards"' in _html()

    def test_stats_bar_total(self):
        assert 'id="diaryTotal"' in _html()

    def test_stats_bar_categories(self):
        assert 'id="diaryCategories"' in _html()

    def test_stats_bar_showing(self):
        assert 'id="diaryShown"' in _html()

    def test_stats_bar_pinned(self):
        assert 'id="diaryPinned"' in _html()

    def test_search_input(self):
        assert 'id="diarySearch"' in _html()

    def test_category_select(self):
        assert 'id="diaryCat"' in _html()

    def test_event_type_select(self):
        assert 'id="diaryEventType"' in _html(), "Event type filter select must exist"

    def test_sort_select(self):
        assert 'id="diarySort"' in _html(), "Sort select must exist"

    def test_importance_range(self):
        assert 'id="diaryImportance"' in _html()

    def test_pinned_only_checkbox(self):
        assert 'id="diaryPinnedOnly"' in _html(), "Pinned-only checkbox must exist"

    def test_browse_recent_button(self):
        assert "diaryBrowseRecent()" in _html(), "Browse Recent button must call diaryBrowseRecent()"

    def test_load_more_button(self):
        assert 'id="diaryLoadMore"' in _html(), "Load More button must exist"

    def test_load_more_calls_loadmore(self):
        assert "loadMoreDiary()" in _html()


# ── JavaScript Functions ───────────────────────────────────────────────────────

class TestJavaScriptFunctions:
    def test_load_diary_stats_defined(self):
        assert "async function loadDiaryStats()" in _html()

    def test_search_diary_defined(self):
        assert "async function searchDiary()" in _html()

    def test_diary_browse_recent_defined(self):
        assert "async function diaryBrowseRecent()" in _html()

    def test_load_more_diary_defined(self):
        assert "async function loadMoreDiary()" in _html()

    def test_fetch_and_render_defined(self):
        assert "async function _fetchAndRenderDiary()" in _html()

    def test_render_diary_cards_defined(self):
        assert "function _renderDiaryCards(" in _html()

    def test_diary_date_group_defined(self):
        assert "function _diaryDateGroup(" in _html()

    def test_diary_expand_defined(self):
        assert "function _diaryExpand(" in _html()

    def test_recent_endpoint_called(self):
        assert "/api/memories/recent" in _html()

    def test_date_groups_today_yesterday(self):
        html = _html()
        assert "'Today'" in html
        assert "'Yesterday'" in html
        assert "'This Week'" in html

    def test_sort_options_in_js(self):
        html = _html()
        assert "'importance'" in html
        assert "'accessed'" in html
        assert "'pinned'" in html

    def test_pinned_filter_applied(self):
        assert "r.pinned" in _html()

    def test_importance_bar_rendered(self):
        # Importance bar is a percentage-width div
        assert "impPct" in _html()

    def test_emotion_badge_rendered(self):
        assert "emotionColor" in _html()

    def test_trust_tier_rendered(self):
        assert "trustColor" in _html()

    def test_expand_collapse_button(self):
        assert "_diaryExpand(" in _html()
        assert "isLong" in _html()

    def test_event_type_filter_applied(self):
        assert "eventType" in _html()

    def test_stats_populates_event_type_select(self):
        html = _html()
        assert "diaryEventType" in html[html.find("async function loadDiaryStats"):][:600]

    def test_load_more_increments_top_k(self):
        html = _html()
        block = html[html.find("async function loadMoreDiary"):][:200]
        assert "_diaryTopK" in block

    def test_dompurify_used_in_cards(self):
        html = _html()
        assert "DOMPurify.sanitize" in html[html.find("function _renderDiaryCards"):][:4000]


# ── Diary Sort Options in HTML ─────────────────────────────────────────────────

class TestSortOptions:
    def test_sort_most_recent_option(self):
        assert "Most Recent" in _html()

    def test_sort_importance_option(self):
        assert "Highest Importance" in _html()

    def test_sort_accessed_option(self):
        assert "Most Accessed" in _html()

    def test_sort_pinned_option(self):
        assert "Pinned First" in _html()
