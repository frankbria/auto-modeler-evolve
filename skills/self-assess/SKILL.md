---
name: self-assess
description: Analyze the project against vision.md and spec.md to find gaps, bugs, and improvement opportunities
tools: [bash, read_file, list_files, search]
---

# Self-Assessment

You are assessing the project against its vision and specification.

## Process

1. **Read vision.md and spec.md** completely
2. **Read the current project state** — list files, read key modules
3. **Compare spec features vs implementation**:
   - Which features from spec.md are implemented?
   - Which are partially done?
   - Which haven't been started?
4. **Try using the project.** Run it, test edge cases:
   - Does it start without errors?
   - Do the implemented features work as specified?
   - Are there crashes, bad error messages, or missing functionality?
5. **Check JOURNAL.md.** Have you tried something before that failed?

## What to look for

- Missing features from spec.md — prioritize by spec order
- Broken functionality — things that used to work but don't
- Missing tests — features without test coverage
- Missing error handling — silent failures, unhelpful messages
- Missing edge cases — empty input, invalid data, boundary conditions
- UX gaps — confusing behavior, unclear output

## Output

Write findings as a prioritized list:

```
SELF-ASSESSMENT Day [N]:
Spec coverage: X of Y features implemented

1. [CRITICAL/HIGH/MEDIUM/LOW] Description
2. ...

Next priorities:
- Feature X from spec (highest unimplemented)
- Bug Y discovered during testing
```

Then decide what to tackle this session.
