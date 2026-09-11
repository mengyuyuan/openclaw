# Duplicate replay evidence captures

Screenshots for the fix that resolves same-run live replays by identical display text
(duplicate assistant replies in the terminal UI).

- `before.png` — the affected terminal session: the reply pair is rendered twice (red box = the duplicate copy).
- `after.png` — after the fix: the reply renders once (green box). One line is redacted.

Captured on Windows in the terminal UI during local verification of the change.
