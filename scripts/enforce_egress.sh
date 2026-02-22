#!/usr/bin/env sh
set -eu

NET_CIDR="${SOVEREIGN_NET_CIDR:-172.20.0.0/16}"
TOOL_GATE_IP="${TOOL_GATE_IP:-172.20.0.3}"

if command -v ufw >/dev/null 2>&1; then
  ufw --force deny out from "$NET_CIDR"
  ufw --force allow out from "$TOOL_GATE_IP" to any port 80 proto tcp
  ufw --force allow out from "$TOOL_GATE_IP" to any port 443 proto tcp
  echo "Applied UFW egress policy for sovereign net"
  exit 0
fi

if command -v iptables >/dev/null 2>&1; then
  iptables -A OUTPUT -s "$NET_CIDR" -j DROP
  iptables -I OUTPUT -s "$TOOL_GATE_IP" -p tcp --dport 80 -j ACCEPT
  iptables -I OUTPUT -s "$TOOL_GATE_IP" -p tcp --dport 443 -j ACCEPT
  echo "Applied iptables egress policy for sovereign net"
  exit 0
fi

echo "Neither ufw nor iptables found" >&2
exit 1
