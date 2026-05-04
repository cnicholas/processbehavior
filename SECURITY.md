# Security Policy

## Supported versions

processbehavior is in **alpha** (0.1.x). Only the most recent 0.1.x release
receives security fixes. When a 0.2.0 ships, 0.1.x will no longer be patched.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email **4948774+cnicholas@users.noreply.github.com** with:

- a description of the issue
- a minimal reproducer (input DataFrame, code snippet, or attached file)
- the version of processbehavior, pandas, and Python you used
- any disclosure timeline you would like

You can expect:

- an acknowledgement within **5 business days**
- a fix or mitigation plan within **30 days** for confirmed issues
- coordinated disclosure: a CVE and credit (if you want it) when the fix
  ships

## Scope

In scope:

- code execution from data ingestion paths (`pd.read_csv`, `pd.read_excel`,
  `result.to_excel`, etc.) when called with hostile input
- silent miscalculation of control limits, residuals, or signals that
  produces results an analyst would mistake for correct output
- supply-chain issues in `processbehavior`'s direct runtime dependencies

Out of scope:

- vulnerabilities in transitive dev/test dependencies that are not exposed
  by the published wheel (please report those upstream)
- denial of service from supplying very large DataFrames — the library
  trusts the analyst's own data

## Out-of-band releases

Critical fixes ship as patch releases (0.1.N+1) outside the normal cadence.
They are noted in `CHANGELOG.md` under a **Security** subheading.
