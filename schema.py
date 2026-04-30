"""Pydantic models shared by bench_latency.py and the analysis scripts.

Event/output models use extra="allow" so server-supplied fields (and any
forward-compat additions) round-trip verbatim. Config is strict (extra="forbid")
and validates the user-supplied config file.

Streaming response shapes mirror OpenAI's Chat Completions streaming API:
  https://platform.openai.com/docs/api-reference/chat/streaming
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Config (strict) + run sidecar
# ---------------------------------------------------------------------------

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: str
    base_url: str
    api_key: str
    num_queries: int
    concurrency: list[int]
    model: str
    extra_body: dict[str, Any]
    timeout: float
    seed: int
    output_dir: str  # parent dir; the bench creates a UUID-named subfolder
                     # inside it and writes result.json + meta.json there.
    metadata: dict[str, Any]  # free-form: server hardware, notes, run tags, etc.


class RunMeta(_Loose):
    started_unix: float
    message: str  # free-form description, e.g. "sglang with speculative decoding"
    config: Config


# ---------------------------------------------------------------------------
# OpenAI Chat Completions: request + streaming response shapes
# ---------------------------------------------------------------------------

class ChatCompletionRequest(_Loose):
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    stream_options: dict[str, Any]


class Usage(_Loose):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # *_details are only present on some servers / token types.
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class ChoiceDelta(_Loose):
    # In streaming, each delta typically carries one of these — role on the
    # first chunk, content on token chunks, tool_calls on tool-call chunks,
    # refusal when the model refuses. The empty-delta finish chunk has none set.
    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamChoice(_Loose):
    index: int
    delta: ChoiceDelta
    # finish_reason is null mid-stream and set on the final chunk for the choice.
    finish_reason: str | None = None
    # logprobs is null unless the request opted in.
    logprobs: dict[str, Any] | None = None


class ChatCompletionChunk(_Loose):
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]
    # usage is null on every chunk except the final one (when stream_options
    # include_usage is true). See FinalChatCompletionChunk for the narrowed
    # type used when we know we're holding the last chunk.
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class FinalChatCompletionChunk(ChatCompletionChunk):
    """The last chunk in a stream. Bench always sets stream_options.include_usage,
    so the final chunk's usage block is populated — narrowed to non-null here so
    consumers don't need a None-guard."""
    usage: Usage


# ---------------------------------------------------------------------------
# Bench events. Every t_s is relative to start-of-program (a single global t0).
# ---------------------------------------------------------------------------

class RequestEvent(_Loose):
    event: Literal["request"] = "request"
    t_s: float
    request_id: int
    concurrency: int
    body: ChatCompletionRequest


class ChunkEvent(_Loose):
    event: Literal["chunk"] = "chunk"
    t_s: float
    request_id: int
    chunk: ChatCompletionChunk


class FinalEvent(_Loose):
    """A request completed successfully. The final chunk is attached and its
    usage block is guaranteed populated."""
    event: Literal["final"] = "final"
    t_s: float
    request_id: int
    chunk: FinalChatCompletionChunk
    status: int


class ErrorEvent(_Loose):
    """A request failed. Any partial chunks were already emitted as ChunkEvents.
    `status` is None if the failure happened before an HTTP status was received
    (e.g., connection refused, DNS, timeout)."""
    event: Literal["error"] = "error"
    t_s: float
    request_id: int
    status: int | None
    error: str


LogEvent = Union[RequestEvent, ChunkEvent, FinalEvent, ErrorEvent]

_DiscriminatedLogEvent = Annotated[LogEvent, Field(discriminator="event")]
LOG_EVENT_ADAPTER: TypeAdapter[LogEvent] = TypeAdapter(_DiscriminatedLogEvent)


def iter_events(path: str) -> Iterator[LogEvent]:
    with open(path) as f:
        for line in f:
            yield LOG_EVENT_ADAPTER.validate_json(line)
