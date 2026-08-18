# Notice / Attribution

This repository is a mirror of the official SEMA implementation released by
Microsoft at **https://github.com/microsoft/SEMA**.

- **Upstream repository:** https://github.com/microsoft/SEMA
- **Synced from upstream commit:** `40ce0473b84c1d9dcf63d7114148ef45311c834b` (`main`, 2026-08-18)
- **Upstream license:** MIT License, Copyright (c) 2026 Microsoft — see [LICENSE](LICENSE)

The source code in `sema/`, `scripts/`, `docs/`, and `examples/` originates from the
upstream repository and is redistributed here under the terms of the MIT License.
The original copyright notices in [LICENSE](LICENSE) and in the individual source file
headers are retained unchanged, as required by that license.

Portions of `sema/trainer/trainer.py` derive from the HuggingFace TRL project,
Copyright 2020-2025 The HuggingFace Team, licensed under the Apache License 2.0.

## Differences from upstream

This mirror is not affiliated with, endorsed by, or maintained by Microsoft.
The following Microsoft-specific items are intentionally not carried over:

- `SECURITY.md` — Microsoft's internal vulnerability-reporting policy (https://aka.ms/SECURITY.md).
  It does not apply to this repository. Please report security issues through the
  upstream repository instead.
- Microsoft-internal entries in `.gitignore` (`.env_ms`, `.env_azure`).

For the authoritative version of this code, and for issues, pull requests, and official
contacts, please use the upstream repository.
