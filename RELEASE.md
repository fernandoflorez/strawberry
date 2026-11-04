---
release type: patch
---

This release improves the performance of argument conversion by computing the
GraphQL name of each argument once per name converter instead of on every
field execution.
