"""Actuator catalog — every side-effecting surface in the system.

Derived from the UH-7 actuator audit.  Each entry is placed in the
migration tier that matches the roadmap's ascending-risk ordering:

  1 read-only data retrieval
  2 isolated local/test operations
  3 document and browser reads
  4 notifications/draft creation
  5 file mutations
  6 calendar/external messages
  7 recovery/admin operations
  8 financial/destructive/public/self-modifying operations

``legacy_path`` names the direct call path that must be disabled before
the actuator can be marked VERIFIED.  An entry with ``legacy_path=None``
has no pre-existing direct path to retire.
"""
from __future__ import annotations

from typing import List

from common.contracts.base import Principal, RiskTier
from common.actuator_registry.registry import (
    ActuatorRegistration,
    ActuatorRegistry,
    MigrationTier,
)


def _reg(
    identity: str,
    display_name: str,
    description: str,
    risk_tier: RiskTier,
    migration_tier: MigrationTier,
    action_types: List[str],
    reversible: bool = False,
    legacy_path: str | None = None,
) -> ActuatorRegistration:
    return ActuatorRegistration(
        identity=identity,
        display_name=display_name,
        description=description,
        risk_tier=risk_tier,
        migration_tier=migration_tier,
        action_types=action_types,
        reversible=reversible,
        legacy_path=legacy_path,
    )


# ── Tier 1: read-only data retrieval ─────────────────────────────────

TIER_1_READ_ONLY: List[ActuatorRegistration] = [
    _reg("broker-bridge", "Broker Bridge",
         "Binance/yfinance market and account reads",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["market_ticker_read", "account_balance_read", "orderbook_read"],
         reversible=True),
    _reg("alpha-signals", "Alpha Signals",
         "Quantitative futures signals from public endpoints",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["alpha_signal_read"], reversible=True),
    _reg("market-data", "Market Data",
         "Price and volume retrieval",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["market_data_read"], reversible=True),
    _reg("docker-watcher", "Docker Watcher",
         "Container health polling",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["container_status_read"], reversible=True),
    _reg("git-watcher", "Git Watcher",
         "Repository status polling",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["git_status_read"], reversible=True),
    _reg("email-reader", "Email Reader",
         "IMAP read-only inbox polling",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["email_read"], reversible=True),
    _reg("calendar-service", "Calendar Service",
         "CalDAV event polling",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["calendar_read"], reversible=True),
    _reg("news-feed", "News Feed",
         "RSS aggregation",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["news_read"], reversible=True),
    _reg("sysmetrics", "System Metrics",
         "CPU/memory/disk snapshot",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["system_metrics_read"], reversible=True),
    _reg("weather-service", "Weather Service",
         "Weather summary retrieval",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["weather_read"], reversible=True),
    _reg("service-watchdog", "Service Watchdog",
         "Component health checks",
         RiskTier.OBSERVE, MigrationTier.READ_ONLY,
         ["service_health_read"], reversible=True),
]


# ── Tier 2: isolated local/test operations ───────────────────────────

TIER_2_LOCAL_TEST: List[ActuatorRegistration] = [
    _reg("shell-sandbox", "Shell Sandbox",
         "Allowlisted read-only shell commands in a sandbox",
         RiskTier.PROPOSE, MigrationTier.LOCAL_TEST,
         ["sandbox_shell_read"], reversible=True),
    _reg("executor-sandbox", "Executor Sandbox",
         "AST-validated Python expression evaluation and noop tools",
         RiskTier.PROPOSE, MigrationTier.LOCAL_TEST,
         ["sandbox_python_eval", "noop"], reversible=True),
]


# ── Tier 3: document and browser reads ───────────────────────────────

TIER_3_DOCUMENT_READ: List[ActuatorRegistration] = [
    _reg("web-scout", "Web Scout",
         "DuckDuckGo search and URL fetch for information gathering",
         RiskTier.PROPOSE, MigrationTier.DOCUMENT_READ,
         ["web_search", "web_fetch"], reversible=True),
    _reg("document-parser", "Document Parser",
         "Document text and structure extraction",
         RiskTier.PROPOSE, MigrationTier.DOCUMENT_READ,
         ["document_parse"], reversible=True),
    _reg("browser-reader", "Browser Reader",
         "Playwright page scrape and screenshot — read-only surface",
         RiskTier.PROPOSE, MigrationTier.DOCUMENT_READ,
         ["browser_scrape", "browser_screenshot"], reversible=True,
         legacy_path="browser-agent:/navigate"),
    _reg("screen-capture", "Screen Capture",
         "Screenshot capture and OCR extraction",
         RiskTier.PROPOSE, MigrationTier.DOCUMENT_READ,
         ["screen_capture", "screen_ocr"], reversible=True),
]


# ── Tier 4: notifications / draft creation ───────────────────────────

TIER_4_NOTIFICATION: List[ActuatorRegistration] = [
    _reg("notify-service", "Notify Service",
         "Desktop notification dispatch",
         RiskTier.ACT_SUPERVISED, MigrationTier.NOTIFICATION,
         ["notify_desktop"], reversible=False,
         legacy_path="notify-service:/notify (unauthenticated)"),
    _reg("tts-service", "TTS Service",
         "Text-to-speech audio synthesis",
         RiskTier.ACT_SUPERVISED, MigrationTier.NOTIFICATION,
         ["tts_speak"], reversible=False),
    _reg("monitor-service", "Monitor Service",
         "Rule-based alerting and rule CRUD",
         RiskTier.ACT_SUPERVISED, MigrationTier.NOTIFICATION,
         ["monitor_alert", "monitor_rule_write"], reversible=False,
         legacy_path="monitor-service:/rules (unauthenticated CRUD)"),
    _reg("screen-watcher", "Screen Watcher",
         "Screenshot diff alerting",
         RiskTier.ACT_SUPERVISED, MigrationTier.NOTIFICATION,
         ["screen_change_alert"], reversible=False),
]


# ── Tier 5: file mutations ───────────────────────────────────────────

TIER_5_FILE_MUTATION: List[ActuatorRegistration] = [
    _reg("vault-sync", "Vault Sync",
         "Obsidian vault markdown export",
         RiskTier.ACT_SUPERVISED, MigrationTier.FILE_MUTATION,
         ["vault_export", "vault_ingest"], reversible=False,
         legacy_path="vault-sync:/export"),
    _reg("checkpoint-manager", "Checkpoint Manager",
         "System state snapshot and rollback",
         RiskTier.ACT_SUPERVISED, MigrationTier.FILE_MUTATION,
         ["checkpoint_create", "checkpoint_restore"], reversible=False,
         legacy_path="agentic:/checkpoint/{id}/restore (unauthenticated)"),
    _reg("memory-writer", "Memory Writer",
         "memu-core memory persistence",
         RiskTier.ACT_SUPERVISED, MigrationTier.FILE_MUTATION,
         ["memory_write"], reversible=False),
]


# ── Tier 6: calendar / external messages ─────────────────────────────

TIER_6_EXTERNAL_MESSAGE: List[ActuatorRegistration] = [
    _reg("telegram-bot", "Telegram Bot",
         "Outbound Telegram message and alert dispatch",
         RiskTier.ACT_SUPERVISED, MigrationTier.EXTERNAL_MESSAGE,
         ["telegram_send", "telegram_alert"], reversible=False,
         legacy_path="telegram-bot:/alert (unauthenticated)"),
    _reg("calendar-writer", "Calendar Writer",
         "CalDAV event creation and modification",
         RiskTier.ACT_SUPERVISED, MigrationTier.EXTERNAL_MESSAGE,
         ["calendar_write"], reversible=False),
]


# ── Tier 7: recovery / admin operations ──────────────────────────────

TIER_7_RECOVERY_ADMIN: List[ActuatorRegistration] = [
    _reg("supervisor", "Supervisor",
         "Service recovery actions and circuit-breaker control",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.RECOVERY_ADMIN,
         ["service_recover", "breaker_reset"], reversible=False),
    _reg("heartbeat", "Heartbeat",
         "Auto-sleep: memory compression and decay",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.RECOVERY_ADMIN,
         ["auto_sleep", "memory_compress", "memory_decay"], reversible=False),
    _reg("backup-service", "Backup Service",
         "Database and memory backup creation",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.RECOVERY_ADMIN,
         ["backup_create"], reversible=True,
         legacy_path="backup-service:/backup (unauthenticated)"),
    _reg("ledger-worker", "Ledger Worker",
         "Audit-chain verification and archival",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.RECOVERY_ADMIN,
         ["ledger_archive", "ledger_verify"], reversible=True),
]


# ── Tier 8: financial / destructive / self-modifying ─────────────────

TIER_8_FINANCIAL_DESTRUCTIVE: List[ActuatorRegistration] = [
    _reg("paper-trader", "Paper Trader",
         "Simulated position open/close against the paper ledger",
         RiskTier.ACT_SUPERVISED, MigrationTier.FINANCIAL_DESTRUCTIVE,
         ["paper_trade_open", "paper_trade_close"], reversible=True,
         legacy_path="strategy_engine:auto_trade()"),
    _reg("executor-shell", "Executor Shell",
         "Allowlisted shell and script execution",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.FINANCIAL_DESTRUCTIVE,
         ["shell_exec", "script_exec"], reversible=False,
         legacy_path="executor:/execute"),
    _reg("browser-actor", "Browser Actor",
         "Playwright click and form-fill — mutating web interaction",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.FINANCIAL_DESTRUCTIVE,
         ["browser_click", "browser_type"], reversible=False,
         legacy_path="browser-agent:/click,/type (unauthenticated)"),
    _reg("db-restore", "Database Restore",
         "PostgreSQL restore from backup — overwrites live state",
         RiskTier.ACT_AUTONOMOUS, MigrationTier.FINANCIAL_DESTRUCTIVE,
         ["db_restore"], reversible=False,
         legacy_path="backup-service:/restore/postgres (unauthenticated)"),
]


ALL_ACTUATORS: List[ActuatorRegistration] = (
    TIER_1_READ_ONLY
    + TIER_2_LOCAL_TEST
    + TIER_3_DOCUMENT_READ
    + TIER_4_NOTIFICATION
    + TIER_5_FILE_MUTATION
    + TIER_6_EXTERNAL_MESSAGE
    + TIER_7_RECOVERY_ADMIN
    + TIER_8_FINANCIAL_DESTRUCTIVE
)


def build_catalog(principal: Principal) -> ActuatorRegistry:
    """Build a registry populated with the full actuator catalog.

    Every actuator starts at ``MigrationStatus.LEGACY``.  Callers advance
    each one explicitly through disable_legacy_path → MIGRATING →
    VERIFIED → ACTIVE.
    """
    registry = ActuatorRegistry(principal=principal)
    for entry in ALL_ACTUATORS:
        registry.register(_reg(
            identity=entry.identity,
            display_name=entry.display_name,
            description=entry.description,
            risk_tier=entry.risk_tier,
            migration_tier=entry.migration_tier,
            action_types=entry.action_types,
            reversible=entry.reversible,
            legacy_path=entry.legacy_path,
        ))
    return registry
