---
name: Discord bot runtime
description: Runtime and secret-handling constraints for the Battle of Creation Discord bot.
---

The Discord bot must run as a continuously running console workflow rather than an HTTP-only or sleeping process, and its configured secret name must match the name used by the runtime.

**Why:** Discord Gateway sessions disconnect when the process is stopped, and a secret-name mismatch looks like an invalid or missing token even when the token itself is valid.

**How to apply:** When changing startup or deployment settings, verify the workflow stays running and keep the canonical token key documented alongside the legacy fallback.