import json
from datetime import datetime

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.orchestrator import build_system_prompt
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
    statement = select(Conversation).where(
        Conversation.project_id == project_id
    )
    conversation = session.exec(statement).first()
    if not conversation:
        conversation = Conversation(project_id=project_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    # Load existing messages
    messages = json.loads(conversation.messages)

    # Append user message
    messages.append(
        {
            "role": "user",
            "content": body.message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Get dataset for context
    dataset_stmt = select(Dataset).where(Dataset.project_id == project_id)
    dataset = session.exec(dataset_stmt).first()

    system_prompt = build_system_prompt(project, dataset)

    # Build API messages (without timestamps for Claude)
    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    client = anthropic.Anthropic()

    # We need to save conversation after streaming completes, so we capture
    # the full response text and save it in a finally-style wrapper.
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

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            # Save assistant response to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            # Use a fresh session to save since the original may be closed
            from db import engine

            with Session(engine) as save_session:
                conv = save_session.get(Conversation, conversation.id)
                if conv:
                    conv.messages = json.dumps(messages)
                    conv.updated_at = datetime.utcnow()
                    save_session.add(conv)
                    save_session.commit()

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
    statement = select(Conversation).where(
        Conversation.project_id == project_id
    )
    conversation = session.exec(statement).first()
    if not conversation:
        return {"messages": []}

    return {"messages": json.loads(conversation.messages)}
