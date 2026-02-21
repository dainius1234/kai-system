#!/usr/bin/env bash
set -euo pipefail

PORT=$(shuf -i 1024-65535 -n1)
if [[ -f .env ]]; then
  if grep -q '^PORT=' .env; then
    sed -i "s/^PORT=.*/PORT=${PORT}/" .env
  else
    echo "PORT=${PORT}" >> .env
  fi
else
  echo "PORT=${PORT}" > .env
fi

if [[ -f /etc/ssh/sshd_config ]]; then
  if grep -q '^Port ' /etc/ssh/sshd_config; then
    sed -i "s/^Port .*/Port ${PORT}/" /etc/ssh/sshd_config
  else
    echo "Port ${PORT}" >> /etc/ssh/sshd_config
  fi
  systemctl restart ssh || systemctl restart sshd || true
fi

echo "SSH port set to ${PORT}"
