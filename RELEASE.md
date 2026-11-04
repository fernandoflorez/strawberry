---
release type: patch
---

This release improves the performance of resolving fields without field
extensions by calling the resolver directly instead of building the extension
chain on every call.
