import json
from typing import Optional

from models.dataset import Dataset
from models.project import Project


def build_system_prompt(
    project: Project, dataset: Optional[Dataset] = None
) -> str:
    """Build a Claude system prompt with project and dataset context.

    The prompt positions the assistant as a helpful data analyst colleague
    who speaks in plain English and explains technical terms when used.
    """
    parts = [
        "You are a helpful data analyst colleague working on the project "
        f'"{project.name}".',
    ]

    if project.description:
        parts.append(f"Project description: {project.description}")

    parts.append(
        "Speak in plain English. When you use a technical term, briefly "
        "explain what it means in parentheses. Suggest next steps when "
        "appropriate. Focus on actionable insights."
    )

    if dataset:
        parts.append(f"\nThe user has uploaded a dataset: {dataset.filename}")
        parts.append(f"Rows: {dataset.row_count}, Columns: {dataset.column_count}")

        if dataset.columns:
            try:
                columns = json.loads(dataset.columns)
                col_descriptions = []
                for col in columns:
                    desc = f"  - {col['name']} ({col['dtype']})"
                    if "mean" in col and col["mean"] is not None:
                        desc += f", mean={col['mean']:.2f}"
                    if col.get("null_pct", 0) > 0:
                        desc += f", {col['null_pct']}% missing"
                    col_descriptions.append(desc)
                parts.append("Columns:\n" + "\n".join(col_descriptions))
            except (json.JSONDecodeError, KeyError):
                pass

        if dataset.profile:
            try:
                profile = json.loads(dataset.profile)
                parts.append(
                    f"Dataset profile summary: {json.dumps(profile, default=str)}"
                )
            except json.JSONDecodeError:
                pass
    else:
        parts.append(
            "\nNo dataset has been uploaded yet. Help the user get started by "
            "asking them to upload a CSV file."
        )

    return "\n\n".join(parts)
