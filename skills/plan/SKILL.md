---
name: plan
description: Plan project structure and implementation strategy from vision.md and spec.md
tools: [bash, read_file, write_file, list_files]
---

# Planning

You are planning a project from scratch based on vision.md and spec.md.

## When to use this skill

- Day 0 (bootstrap) — plan the entire project structure
- Before implementing a complex feature — plan the approach
- When multiple spec features interact — plan the integration

## Process

1. **Read vision.md** — understand the "why" and "what"
2. **Read spec.md** — understand the "how": tech stack, architecture, features
3. **Identify constraints**:
   - What tech stack is specified?
   - What's the deployment target?
   - What testing strategy is required?
4. **Plan the structure**:
   - Directory layout
   - Key files and their responsibilities
   - Dependency graph between features
5. **Determine build order**:
   - What can be built first with no dependencies?
   - What depends on what?
   - What's the minimum viable feature set?

## Output

Write a clear implementation plan:

```
PROJECT PLAN — Day [N]

Structure:
  project/
  ├── [dir]/ — [purpose]
  └── [file] — [purpose]

Build Order:
1. [Feature] — [why first]
2. [Feature] — [depends on #1]
3. ...

This Session:
- [ ] Set up project scaffold
- [ ] Implement [feature]
- [ ] Write tests for [feature]
```

## Rules

- Follow the spec's tech stack exactly. Don't substitute technologies.
- Plan for testability from the start.
- Keep the structure as simple as possible — add complexity only when needed.
- Don't over-plan. Plan what you'll build THIS session, sketch the rest.
