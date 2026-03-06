# Kai Design System

> Why Apple/Ubuntu feel premium and how Kai achieves the same.

## Why Apple Feels Better Than Windows — The Actual Reasons

### 1. Consistency (The #1 Difference)

**Apple:** Every element shares the same corner radius (13px on iOS, varies by context on macOS), the same shadow language, the same blur amount, the same animation curve. You never see a sharp-cornered dialog next to a rounded button. The system font (SF Pro) is the same everywhere. Spacing follows a strict 8-point grid.

**Windows:** Mixed metaphors. Fluent Design (rounded, translucent), Win32 (sharp, flat), legacy windows all visible simultaneously. Three different scrollbar styles in one OS. Settings use Segoe UI; Command Prompt uses Consolas. No single source of truth.

**Lesson for Kai:** One CSS file. One font. One border-radius. One shadow scale. One spacing grid. No exceptions. A card in Chat looks identical to a card in Thinking.

### 2. Animation Physics (The Subconscious Feeling)

**Apple:** Every animation follows spring physics — elements overshoot slightly then settle. Duration: 300-350ms. Curve: `cubic-bezier(0.25, 0.46, 0.45, 0.94)` (ease-out with slight spring). Nothing teleports. Nothing moves linearly. Scrolling has momentum. Dragging has inertia.

**Ubuntu/GNOME:** Similar but simplified — `ease-in-out` at 200-250ms. Clean and purposeful. No bounce, but never instant. Activities overview zooms, windows slide. Every animation tells you "something moved from A to B."

**Windows:** Inconsistent. Some animations are 100ms (too fast to read), others 500ms (feels sluggish). Flyouts appear without transition. Start menu tiles have a different animation language from notification center.

**Lesson for Kai:** 
- Page transitions: 300ms, `cubic-bezier(0.4, 0, 0.2, 1)` (Material's standard curve — proven)
- Micro-interactions (hover, press): 150ms, `ease-out`
- Data loading: 500ms shimmer at `ease-in-out`
- Nothing is instant. Nothing takes more than 400ms.

### 3. Depth and Layering (Why Things "Pop")

**Apple:** Uses real-world metaphors — sheets slide over content (not replacing it), modals dim the background, sidebars cast shadows. Everything has a z-position. Backdrop blur (`-webkit-backdrop-filter: blur(20px)`) makes UI elements feel like frosted glass floating over content. macOS Sonoma uses this everywhere.

**Ubuntu:** Similar but with more opacity, less blur. Dark theme uses subtle elevation — cards feel like they're stacked paper. GNOME Shell uses `box-shadow` layers for depth.

**Windows:** Acrylic material (blur) exists but inconsistently applied. Many surfaces are flat-opaque, giving no sense of hierarchy.

**Lesson for Kai:**
- Background: solid dark `#0a0a0c`  
- Surface 1 (sidebar): `rgba(20, 20, 25, 0.8)` + `backdrop-filter: blur(20px)` + `border: 1px solid rgba(255,255,255,0.06)`
- Surface 2 (cards): `rgba(28, 28, 36, 0.6)` + `backdrop-filter: blur(12px)` + softer border
- Surface 3 (elevated/modals): `rgba(36, 36, 48, 0.9)` + `blur(24px)` + stronger shadow
- Every layer is slightly transparent so the depth is visible

### 4. Typography (The Invisible Perfection)

**Apple SF Pro:** Optimised for screens. Variable weight. Tight but readable. Consistent metrics across sizes. Apple never uses more than 3 weights on one screen (Regular, Medium, Bold).

**Key principle:** Type hierarchy is achieved through SIZE and WEIGHT, not color or decoration. Headers are bigger + bolder. Body is regular. Captions are smaller + lighter. That's it.

**Lesson for Kai:**
- Font: Inter (closest free equivalent to SF Pro — designed for screens, variable weight)
- Self-hosted for offline operation (one WOFF2 file, ~100KB)
- Scale: 12 / 13 / 15 / 18 / 24 / 32px (1.2 ratio)
- Weights: 400 (body), 500 (labels), 600 (headings), 700 (emphasis only)
- Line-height: 1.5 for body, 1.3 for headings
- Letter-spacing: -0.01em for headings (tight), 0 for body, 0.02em for tiny caps

### 5. Color Restraint (Less Is More)

**Apple:** Dark mode uses exactly 5 semantic colors: label (white), secondary label (gray), accent (blue), success (green), destructive (red). Every other color derives from these. The accent color is configurable but the system still uses only one at a time.

**Ubuntu:** Similar — Aubergine accent, Human (warm) palette, Slate (cool) neutrals. Never more than 2 accent colors visible at once.

**Windows:** Accent color, but also legacy colors, high-contrast overrides, and inconsistent use of the palette across native vs. UWP apps.

**Lesson for Kai:**
- Primary accent: `#00e5ff` (cyan) — for interactive elements, active states, links
- Secondary accent: `#7c4dff` (purple) — for Kai's identity, avatar glow, secondary actions
- Success: `#00c853`, Warning: `#ffc107`, Error: `#ff3d3d`
- Text: `#e0e4ef` (primary), `#9ca3af` (secondary), `#6b7280` (muted)
- **Rule:** No more than 2 accent colors visible in any single viewport. Purple is for Kai's presence, Cyan is for your actions.

### 6. Whitespace (The Courage to Leave Empty Space)

**Apple:** Generous padding. A settings screen might show only 6 items when it could fit 12. Content breathes. You never feel cramped. This signals confidence — "we have so few, important things to show you."

**Windows:** Packs UI tightly. Lists have small row height. Dialog buttons crowd each other. Feels busy, anxious.

**Lesson for Kai:**
- 8px base grid. All spacing is multiples: 8, 16, 24, 32, 48, 64
- Card padding: 24px (not 16)
- Section gaps: 24px minimum
- Touch targets: 44px minimum (Apple's standard)
- Let the dark background breathe — it IS the design

### 7. State Communication (The System Feels Alive)

**Apple:** Battery icon shows charge level. Wi-Fi shows signal strength. Processing shows a progress bar or spinner. Every pixel communicates system state. Menu bar is always current.

**Ubuntu:** Top bar shows time, battery, network, audio. GNOME activities corner glows when there are notifications. System tray is small but always honest.

**Lesson for Kai:**
- The Kai orb avatar IS the system state indicator — always visible, always current
- Orb states map to system states (idle/thinking/speaking/listening/dreaming/error)
- Health status dots in sidebar footer — glanceable without clicking
- Toast notifications for events — information reaches you, you don't hunt for it

---

## Design Tokens

### Colors
```css
:root {
  /* Surfaces */
  --bg:          #0a0a0c;
  --surface:     rgba(20, 20, 25, 0.8);
  --surface2:    rgba(28, 28, 36, 0.6);
  --surface3:    rgba(36, 36, 48, 0.9);
  --border:      rgba(255, 255, 255, 0.06);
  --border-hover: rgba(255, 255, 255, 0.12);

  /* Accents */
  --accent:      #00e5ff;
  --accent-dim:  rgba(0, 229, 255, 0.15);
  --accent2:     #7c4dff;
  --accent2-dim: rgba(124, 77, 255, 0.15);

  /* Semantic */
  --ok:          #00c853;
  --warn:        #ffc107;
  --danger:      #ff3d3d;
  --ok-dim:      rgba(0, 200, 83, 0.12);
  --warn-dim:    rgba(255, 193, 7, 0.12);
  --danger-dim:  rgba(255, 61, 61, 0.12);

  /* Text */
  --text:        #e0e4ef;
  --text2:       #9ca3af;
  --text3:       #6b7280;

  /* Spacing (8px grid) */
  --sp-1: 4px;  --sp-2: 8px;   --sp-3: 12px;
  --sp-4: 16px; --sp-5: 20px;  --sp-6: 24px;
  --sp-8: 32px; --sp-10: 40px; --sp-12: 48px;

  /* Radius */
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 24px;
  --r-full: 9999px;

  /* Shadows (layered for realistic depth) */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.15);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.3), 0 2px 4px rgba(0,0,0,0.2);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.4), 0 6px 10px rgba(0,0,0,0.25);

  /* Motion */
  --ease-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --duration-fast: 150ms;
  --duration-normal: 300ms;
  --duration-slow: 500ms;

  /* Typography */
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'Fira Code', 'Cascadia Code', 'SF Mono', monospace;

  /* Blur */
  --blur-sm: blur(8px);
  --blur-md: blur(16px);
  --blur-lg: blur(24px);

  /* Layout */
  --sidebar-w: 260px;
  --header-h: 56px;
  --content-max: 900px;

  /* Z-index layers */
  --z-sidebar: 10;
  --z-header: 20;
  --z-toast: 100;
  --z-modal: 200;
}
```

### Typography Scale
| Role | Size | Weight | Line-Height | Letter-Spacing |
|------|------|--------|-------------|----------------|
| Display | 32px | 700 | 1.2 | -0.02em |
| Heading | 24px | 600 | 1.3 | -0.01em |
| Title | 18px | 600 | 1.3 | -0.01em |
| Body | 15px | 400 | 1.5 | 0 |
| Body small | 13px | 400 | 1.5 | 0 |
| Caption | 12px | 500 | 1.4 | 0.02em |
| Tiny | 11px | 500 | 1.3 | 0.04em |

---

## Component Library

### Glass Card
```css
.card {
  background: var(--surface2);
  backdrop-filter: var(--blur-md);
  -webkit-backdrop-filter: var(--blur-md);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: var(--sp-6);
  transition: border-color var(--duration-fast) var(--ease-out),
              box-shadow var(--duration-fast) var(--ease-out),
              transform var(--duration-fast) var(--ease-out);
}
.card:hover {
  border-color: var(--border-hover);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
```

### Kai Avatar Orb (6 states)
The orb is a CSS/SVG animated sphere that communicates Kai's state:

| State | Visual | Trigger |
|-------|--------|---------|
| **Idle** | Slow breathing pulse (scale 1.0↔1.03, 4s). Cyan/purple gradient. | Default |
| **Thinking** | Orbital ring spins (1.5s rotation). Glow intensifies. | Streaming response |
| **Listening** | Contracts slightly (scale 0.97). Border brightens. Audio ripple rings. | Voice input active |
| **Speaking** | Sound wave rings pulse outward from center. | TTS playing |
| **Dreaming** | Slow morphing aurora gradient (cyan→purple→cyan). Stars drift. | Dream consolidation |
| **Error** | Single red pulse, then dims to muted orange. Not alarming. | Service down |

### Toast Notifications
Slide in from top-right. Auto-dismiss at 5s. Stack up to 3 visible.
Types: info (cyan border), success (green), warning (amber), error (red).

### Navigation
Sidebar: 260px fixed left. Glass surface. Avatar orb at top, nav items center, health dots at bottom.
On mobile (<768px): sidebar collapses to bottom tab bar (4 items).

---

## Page Architecture

### Before (3 separate HTML files)
```
/static/chat.html     ← separate page, separate CSS, separate header
/static/index.html    ← different font, different variables, different look
/static/thinking.html ← yet another header, shared some vars
```

### After (1 app shell + embedded sections)
```
/static/app.html      ← unified shell
  ├── style.css       ← single design system
  ├── sidebar          (avatar + nav + health)
  ├── #view-chat       (chat section — hidden/shown)
  ├── #view-dashboard  (control panel — hidden/shown)
  ├── #view-thinking   (thinking pathways — hidden/shown)
  └── #view-settings   (future: preferences)
```

Navigation is instant — no page reloads. Click sidebar item → old section fades out (150ms), new section fades in (300ms). URL updates via `history.pushState` so browser back/forward works.

### Routing
- `/` → app shell, default to chat view
- `/chat` → app shell, chat view
- `/control` → app shell, dashboard view  
- `/thinking` → app shell, thinking view
- All API endpoints unchanged (`/api/chat`, `/api/thinking`, `/health`, etc.)

---

## Build Sequence

1. **`style.css`** — Design tokens, reset, typography, glass cards, animations, utilities
2. **`app.html`** — App shell (sidebar + header + content-area + toast container)
3. **Avatar orb** — SVG + CSS animation component (6 states)
4. **Chat section** — Port chat.html JS/HTML into `#view-chat`
5. **Dashboard section** — Port index.html into `#view-dashboard`
6. **Thinking section** — Port thinking.html into `#view-thinking`
7. **Toast system** — Notification component, wire to SSE pubsub
8. **`dashboard/app.py`** — New routes for `/`, `/chat`, `/control`, `/thinking` all serving app.html
9. **Inter font** — Self-hosted WOFF2 for offline operation

---

## Reference Products (What Inspired This)

- **macOS Ventura/Sonoma** — Glass morphism, consistent blur, spring animations
- **GNOME 44/45** — Clean dark mode, minimal chrome, purposeful animation
- **Linear** — Best-in-class dark app UI (glass cards, subtle gradients, tight typography)
- **Raycast** — Command palette UX, fast transitions, glass surfaces
- **Vercel Dashboard** — Clean data presentation, monospace data + sans-serif labels
- **Siri Orb** — Abstract AI presence indicator (not a face, a state)
- **GitHub Copilot Chat** — Streaming text, typing indicator, clean message layout
