---
name: research
description: Search the web and read documentation when implementing unfamiliar things
tools: [bash]
---

# Research

You have internet access through bash. Use it when implementing
something unfamiliar or when you want to see best practices.

## How to search

```bash
curl -s "https://lite.duckduckgo.com/lite?q=your+query" | sed 's/<[^>]*>//g' | head -60
```

## How to read a webpage

```bash
curl -s [url] | sed 's/<[^>]*>//g' | head -100
```

## Rules

- Have a specific question before searching. No aimless browsing.
- Write what you learn to LEARNINGS.md so you never search the same thing twice.
- Read LEARNINGS.md before searching — you may already know the answer.
- Prefer official docs over random blogs.
- When studying other projects, note what's good AND what you'd do differently.

## When to research

- You're implementing something from the spec that you're unfamiliar with
- You hit an error you don't understand
- You want to see best practices for the tech stack specified in spec.md
- A community issue references a concept you don't know
- You're choosing between multiple approaches
