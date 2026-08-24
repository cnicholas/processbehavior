## Summary

- **What:**
- **Why:**
- **Scope:**

## Contract / invariants

Check the ones this change had to keep true. Anything you cannot check, explain below.

- [ ] **Defaults unchanged** — existing calls produce equivalent results
- [ ] **Residuals unaffected** — no residual or design-state recomputation
- [ ] **Row/index alignment preserved** — output row count and ordering unchanged
- [ ] **Output schema compatible** — column names/types and the stats-dict shape
      (`{N, center, lpl, upl}`) still hold
- [ ] **Pinned error strings unchanged** — or the tests that pin them are updated
      deliberately, and the commit says why

## Behaviour changes

<!-- Anything a user could notice. "None" is a fine answer. -->

## Methodology

- [ ] No methodology change — implementation, ergonomics, or docs only
- [ ] Methodology change, validated against Bishop's reference data
      (cite the chapter/equation/table)

## Tests

<!-- Name the test and what it proves, not just that tests were added. -->

- [ ] `pytest tests/ -m "not slow"`
- [ ] `ruff check .`
- [ ] `mypy processbehavior` (advisory — no new errors)
- [ ] Golden masters regenerated deliberately, if the diff touches them

## Notes

<!-- Follow-ups deliberately left out of this PR. -->
