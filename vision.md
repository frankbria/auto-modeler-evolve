# Vision

## Project: AutoModeler

An AI-powered, conversational data modeling platform that turns raw data into deployed
prediction services — no code required. Business analysts upload a spreadsheet, have a
conversation with an AI assistant, and walk away with a validated model running behind
a live API and an interactive dashboard.

### The Problem

Machine learning is powerful but gatekept. A business analyst who knows their data
better than anyone still can't build a model without a data scientist, a weeks-long
backlog, and a deployment pipeline they'll never understand. Existing AutoML tools
(DataRobot, H2O, AutoSklearn) solve the *algorithm* problem but not the *people*
problem — they still assume you know what a "hyperparameter" is.

Meanwhile, the analyst's spreadsheet sits in a shared drive, full of patterns no one
has time to find.

### Who It's For

**Primary:** Business analysts who are data-literate but not code-literate. They
understand their domain deeply (sales patterns, customer behavior, operational
metrics) but hit a wall when it's time to go from "I have a hunch" to "I have proof."

**Secondary:** Small teams without dedicated data science staff who need quick,
reliable predictions — not research papers.

### What AutoModeler Does

AutoModeler meets users where they are: a chat window.

1. **Upload** — Drag in a CSV (or connect a data source). AutoModeler immediately
   shows what it sees: row counts, column types, patterns, anomalies, and plain-English
   summaries ("This looks like monthly sales data with 14 product categories").

2. **Explore** — Ask questions in natural language. "Which products are trending up?"
   "Are there seasonal patterns?" "What's driving returns?" AutoModeler generates
   charts, statistics, and explanations — not code.

3. **Shape** — AutoModeler suggests features: "I notice `order_date` could be split
   into day-of-week and month — seasonal patterns might emerge." The user approves,
   tweaks, or asks "why?" Everything is explained before it's applied.

4. **Model** — Based on the data and the user's goal ("I want to predict next month's
   revenue"), AutoModeler recommends and trains appropriate models. Results are shown as
   comparisons: "Model A is more accurate but Model B is easier to explain to your
   team." The user picks what matters to them.

5. **Validate** — Before anything goes live, AutoModeler walks through what the model
   gets right, what it gets wrong, and where it's uncertain. "This model is 92%
   accurate overall, but struggles with new product categories — here's why."

6. **Deploy** — One click. The model becomes a live API endpoint and an interactive
   prediction dashboard. The analyst can share a link: "Paste in next month's numbers
   and see the forecast." No DevOps, no Docker, no YAML.

### Design Philosophy

- **Conversation over configuration.** Every interaction can happen through chat. Forms
  and buttons exist as shortcuts, not requirements.
- **Explain before executing.** Never silently transform data or train a model. Always
  tell the user what's about to happen and why.
- **Progressive disclosure.** Start simple. Reveal complexity only when the user asks
  for it or when it matters for the decision.
- **Delightful, not just functional.** Smooth transitions, clear visualizations, and
  moments of surprise ("I found something interesting in your data..."). This should
  feel like working with a smart colleague, not operating a machine.

### Success Looks Like

A business analyst uploads their quarterly sales data during a lunch break. By the end
of lunch, they have:
- A clear understanding of what's driving their numbers
- A model that predicts next quarter's revenue by region
- A live dashboard they can share with their VP
- An API their developer can plug into the company's reporting tool

They didn't write a line of code. They didn't read a single documentation page. They
had a conversation.

### What This Is Not

- **Not a notebook replacement.** Data scientists who want fine-grained control should
  use Jupyter. AutoModeler is for the other 90% of the organization.
- **Not a BI tool.** Tableau and Power BI visualize historical data. AutoModeler
  predicts the future.
- **Not a black box.** Every model decision is explainable in plain language. If a user
  asks "why did you pick this model?", they get a real answer.
