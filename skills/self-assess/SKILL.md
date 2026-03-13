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

### Spec Fidelity (Gap Analysis)
- **Claimed vs. actual** — For each [x] item in spec.md, does the implementation
  actually deliver what the spec describes? A passing test doesn't mean the feature
  is complete. Check: does it work from the user's perspective?
- **Shallow implementations** — Features that technically exist but lack depth. E.g.,
  "error handling" that catches exceptions but shows generic messages instead of the
  helpful, specific guidance described in the UX principles.
- **Missing connections** — Features that work in isolation but aren't wired into the
  chat flow as the spec intends. The vision says "chat is the primary interface" — can
  every feature be triggered through conversation?

### Quality Audit
- **Test coverage gaps** — Run coverage reports. Which modules are below 90%? Which
  have zero coverage? Which have tests that only cover the happy path?
- **E2E coverage** — Map each spec phase to Playwright tests. Which user journeys
  have no E2E test? Which have tests that skip critical steps?
- **Integration boundaries** — Are there untested boundaries between backend modules?
  Between frontend and backend? Between the chat orchestrator and core logic?
- **Error resilience** — What happens with bad input? Empty CSV? Single row?
  All-null columns? Non-UTF8 encoding? Test these and document failures.

### Existing Issues
- Broken functionality — things that used to work but don't
- Missing error handling — silent failures, unhelpful messages
- Missing edge cases — empty input, invalid data, boundary conditions
- UX gaps — confusing behavior, unclear output

### Vision Alignment (Innovation Opportunities)
- **Re-read vision.md** — What aspects of the vision aren't yet reflected in the code?
  Not just features, but qualities: "delightful," "feels like a smart colleague,"
  "moments of surprise."
- **User journey friction** — Walk through the full flow mentally. Where would a
  business analyst get confused, stuck, or bored? Those are innovation opportunities.
- **External integrations** — What models, data sources, or tools could make the
  platform more useful for the vision's target user?

## Output

Write findings as a prioritized list:

```
SELF-ASSESSMENT Day [N]:
Spec coverage: X of Y features [x] — Z verified working end-to-end
Test coverage: backend X%, frontend X%
E2E scenarios: X of Y critical paths covered

Gap Analysis:
- [SPEC ITEM] — Status: [working | shallow | broken] — [details]

Quality Issues:
1. [CRITICAL/HIGH/MEDIUM/LOW] Description

Innovation Opportunities:
1. [Description] — traces to vision: "[relevant vision quote]"

Recommended Focus This Session:
- Quality: [specific hardening task]
- Innovation: [specific expansion task]
- Balance recommendation: [quality-heavy | innovation-heavy | 50/50] based on recent sessions
```

Then decide what to tackle this session. Check BACKLOG.md for coordination
with other bot instances before committing to a focus area.
