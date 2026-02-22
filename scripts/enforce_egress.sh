#!/usr/bin/env sh
set -eu

NET_CIDR="${SOVEREIGN_NET_CIDR:-172.20.0.0/16}"
TOOL_GATE_IP="${TOOL_GATE_IP:-172.20.0.3}"
TELEGRAM_BRIDGE_IP="${TELEGRAM_BRIDGE_IP:-172.20.0.22}"
TELEGRAM_IPS="${TELEGRAM_IPS:-}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
ALLOW_HTTPS_PORT="${ALLOW_HTTPS_PORT:-443}"
ALLOW_HTTP_PORT="${ALLOW_HTTP_PORT:-80}"

apply_ufw() {
  if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "warning: TELEGRAM_BOT_TOKEN is empty; alert transport may be disabled" >&2
  fi
  ufw --force deny out from "$NET_CIDR"
  ufw --force allow out from "$TOOL_GATE_IP" to any port "$ALLOW_HTTP_PORT" proto tcp
  ufw --force allow out from "$TOOL_GATE_IP" to any port "$ALLOW_HTTPS_PORT" proto tcp
  ufw --force allow out from "$NET_CIDR" to "$TELEGRAM_BRIDGE_IP" port "$ALLOW_HTTP_PORT" proto tcp
  for ip in $(echo "$TELEGRAM_IPS" | tr ',' ' '); do
    [ -n "$ip" ] && ufw --force allow out from "$NET_CIDR" to "$ip" port "$ALLOW_HTTPS_PORT" proto tcp
  done
  echo "Applied UFW egress whitelist (tool-gate + telegram bridge + optional telegram IPs)"
}

apply_iptables() {
  if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "warning: TELEGRAM_BOT_TOKEN is empty; alert transport may be disabled" >&2
  fi
  iptables -A OUTPUT -s "$NET_CIDR" -j DROP
  iptables -I OUTPUT -s "$TOOL_GATE_IP" -p tcp --dport "$ALLOW_HTTP_PORT" -j ACCEPT
  iptables -I OUTPUT -s "$TOOL_GATE_IP" -p tcp --dport "$ALLOW_HTTPS_PORT" -j ACCEPT
  iptables -I OUTPUT -s "$NET_CIDR" -d "$TELEGRAM_BRIDGE_IP" -p tcp --dport "$ALLOW_HTTP_PORT" -j ACCEPT
  for ip in $(echo "$TELEGRAM_IPS" | tr ',' ' '); do
    [ -n "$ip" ] && iptables -I OUTPUT -s "$NET_CIDR" -d "$ip" -p tcp --dport "$ALLOW_HTTPS_PORT" -j ACCEPT
  done
  echo "Applied iptables egress whitelist (tool-gate + telegram bridge + optional telegram IPs)"
}

if command -v ufw >/dev/null 2>&1; then
  apply_ufw
  exit 0
fi

if command -v iptables >/dev/null 2>&1; then
  apply_iptables
  exit 0
fi

echo "Neither ufw nor iptables found" >&2
exit 1
