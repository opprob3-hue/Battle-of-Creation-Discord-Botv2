---
name: Gemini judge runtime
description: Provider behavior affecting grounded Gemini judging.
---

The direct Gemini API can reject older model names for new users and can return quota errors even when the key is valid; the judge must keep a deterministic fallback and treat provider failures as non-fatal.

**Why:** A provider error should never take the Discord bot offline or decide a match from a partial response.

**How to apply:** Keep the model configurable/current, preserve the capability-only fallback, and surface quota failures as configuration or billing issues rather than retrying endlessly.