import json
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.orchestrator import build_system_prompt
from core.query_engine import generate_chart_for_message
from db import get_session
from models.conversation import Conversation
from models.dataset import Dataset
from models.project import Project

router = APIRouter(prefix="/api/chat", tags=["chat"])

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024


class ChatMessage(BaseModel):
    message: str


@router.post("/{project_id}")
def send_message(
    project_id: str,
    body: ChatMessage,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get or create conversation
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        conversation = Conversation(project_id=project_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    messages = json.loads(conversation.messages)

    messages.append(
        {
            "role": "user",
            "content": body.message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Get dataset for context (and for chart generation)
    dataset_stmt = select(Dataset).where(Dataset.project_id == project_id)
    dataset = session.exec(dataset_stmt).first()

    system_prompt = build_system_prompt(project, dataset)

    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    client = anthropic.Anthropic()

    # Capture dataset info for post-stream chart generation
    dataset_file_path: str | None = dataset.file_path if dataset else None
    column_info: list = json.loads(dataset.columns) if (dataset and dataset.columns) else []

    def stream_response():
        full_response = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    chunk = json.dumps({"type": "token", "content": text})
                    yield f"data: {chunk}\n\n"

        finally:
            # Save assistant response
            messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            from db import engine

            with Session(engine) as save_session:
                conv = save_session.get(Conversation, conversation.id)
                if conv:
                    conv.messages = json.dumps(messages)
                    conv.updated_at = datetime.utcnow()
                    save_session.add(conv)
                    save_session.commit()

        # After text stream, opportunistically generate a chart if the
        # message is about data and we have a dataset loaded
        if dataset_file_path:
            try:
                fp = Path(dataset_file_path)
                if fp.exists() and column_info:
                    df = pd.read_csv(fp)
                    chart = generate_chart_for_message(
                        body.message, df, column_info, full_response
                    )
                    if chart:
                        yield f"data: {json.dumps({'type': 'chart', 'chart': chart})}\n\n"
            except Exception:  # noqa: BLE001
                pass  # Charts are nice-to-have; never crash the chat

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{project_id}/history")
def get_history(
    project_id: str,
    session: Session = Depends(get_session),
):
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        return {"messages": []}

    return {"messages": json.loads(conversation.messages)}
