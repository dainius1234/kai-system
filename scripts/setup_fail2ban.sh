#!/usr/bin/env bash
set -euo pipefail

SSH_PORT="${PORT:?PORT env var must be set}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not available; run on Debian/Ubuntu host"
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y fail2ban

cat >/etc/fail2ban/jail.d/sovereign-ssh.local <<JAIL
[sshd]
enabled = true
port = ${SSH_PORT}
maxretry = 3
findtime = 10m
bantime = 1h
JAIL

systemctl enable fail2ban
systemctl restart fail2ban
fail2ban-client status sshd
