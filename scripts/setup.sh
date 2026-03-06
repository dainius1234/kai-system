#!/usr/bin/env bash
# KAI — Sovereign AI  ·  One-command setup script
# Usage: make setup  (or bash scripts/setup.sh)
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m' NC='\033[0m'
ok()   { printf "${G}✓${NC} %s\n" "$1"; }
warn() { printf "${Y}⚠${NC} %s\n" "$1"; }
fail() { printf "${R}✗${NC} %s\n" "$1"; exit 1; }
info() { printf "${B}→${NC} %s\n" "$1"; }

echo ""
echo "  ██╗  ██╗ █████╗ ██╗"
echo "  ██║ ██╔╝██╔══██╗██║"
echo "  █████╔╝ ███████║██║"
echo "  ██╔═██╗ ██╔══██║██║"
echo "  ██║  ██╗██║  ██║██║"
echo "  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  Sovereign AI"
echo ""

# ── Pre-flight checks ───────────────────────────────────────────────

info "Checking prerequisites..."

command -v docker  >/dev/null 2>&1 || fail "Docker not found. Install: https://docs.docker.com/get-docker/"
ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"

command -v docker compose >/dev/null 2>&1 && COMPOSE="docker compose" || {
  command -v docker-compose >/dev/null 2>&1 && COMPOSE="docker-compose" || fail "docker compose not found."
}
ok "docker compose available"

command -v python3 >/dev/null 2>&1 || fail "Python 3 not found."
ok "Python $(python3 --version | awk '{print $2}')"

command -v git >/dev/null 2>&1 || fail "git not found."
ok "git $(git --version | awk '{print $3}')"

echo ""

# ── Generate .env ────────────────────────────────────────────────────

if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    ok ".env created — edit it with your API keys before running"
  else
    info "Creating minimal .env..."
    cat > .env <<'ENVEOF'
# KAI .env — edit these values
KAI_MODE=WORK
TOOL_GATE_HMAC_SECRET=change-me-to-a-random-secret
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
ENVEOF
    ok ".env created — add your API keys before running"
  fi
else
  ok ".env already exists"
fi

echo ""

# ── Build images ─────────────────────────────────────────────────────

info "Building core images (this may take a few minutes on first run)..."
$COMPOSE -f docker-compose.minimal.yml build
ok "Core images built"

echo ""

# ── Syntax check ─────────────────────────────────────────────────────

info "Running syntax check..."
if make go_no_go 2>/dev/null; then
  ok "Syntax check passed"
else
  warn "Syntax check had warnings (non-fatal)"
fi

echo ""

# ── Done ─────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys"
echo "    2. make core-up        — start core services"
echo "    3. Open http://localhost:8050/app"
echo ""
echo "  Useful commands:"
echo "    make core-down         — stop services"
echo "    make test-core         — run test suite"
echo "    make merge-gate        — full validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
