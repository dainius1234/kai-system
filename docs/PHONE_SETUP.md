# Phone Setup — Install Kai as a PWA

Kai ships as a Progressive Web App (PWA). One tap from your phone browser
installs it to your home screen with a full-screen app shell, offline fallback,
and no app-store account required.

## Prerequisites

- The Kai stack must be reachable from your phone (same network, or via Tailscale)
- Know the IP/hostname: `http://<your-host>:8008` (or the reverse-proxy URL)

---

## Android (Chrome)

1. Open Chrome and navigate to `http://<host>:8008/app`
2. Tap the **three-dot menu** (⋮) in the top-right corner
3. Tap **"Add to Home screen"**
4. Confirm the name ("KAI") and tap **"Add"**
5. Kai appears on your home screen — tap to launch in full-screen

> **Tip:** If you don't see "Add to Home screen", open `chrome://flags`, search for
> "Desktop PWA", enable it, and relaunch.

---

## iOS (Safari)

1. Open **Safari** and navigate to `http://<host>:8008/app`
2. Tap the **Share button** (box with upward arrow) at the bottom of the screen
3. Scroll down and tap **"Add to Home Screen"**
4. Edit the name if needed ("KAI") and tap **"Add"**
5. Kai appears on your home screen — tap to launch in standalone mode

> **Note:** PWA install on iOS requires Safari. Chrome/Firefox on iOS do not
> support the "Add to Home Screen" PWA flow.

---

## What you get

| Feature | Supported |
|---|---|
| Offline app shell | Yes — chat UI loads without network |
| Live chat | Requires network to reach Kai's backend |
| Push notifications | Not yet (requires HTTPS + VAPID keys) |
| Background sync | Not yet |
| Voice input | Yes — tap the 🎤 button in chat |

The service worker caches the app shell (`/`, `/app`, CSS, icons, manifest).
API calls (`/api/*`, `/stream`, `/health`) always go to the live server — they
are never intercepted by the cache.

---

## Troubleshooting

**"Add to Home Screen" not shown (Android):**
The page must be served over HTTPS or `localhost`. If you're on a local IP
(`192.168.x.x`), set up a Tailscale hostname or a local HTTPS proxy (e.g.
Caddy with a self-signed cert trusted by the device).

**App installed but shows blank screen offline:**
The service worker needs one successful online load to populate the cache.
Open the app once while connected, then it works offline.

**Tailscale access:**
If Kai runs behind Tailscale, install the Tailscale app on your phone, log in,
and use the MagicDNS hostname (e.g. `http://kai-host:8008/app`).

---

## Updating the app

When Kai is updated, the service worker detects a new cache version and
refreshes automatically on the next page load. You don't need to re-install
— just close and reopen the app.
