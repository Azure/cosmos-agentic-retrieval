"""Tell callers why a model request failed without exposing provider diagnostics.

A generic failure hides whether the caller should retry, change the question, or
contact the operator. Raw SDK exceptions can contain endpoints and credentials.
This module converts them into fixed messages, categories, HTTP statuses and
validated retry hints, which the agent loop carries to the HTTP endpoint.
It does not retry requests; the OpenAI SDK owns that policy in ChatClient.
"""

from __future__ import annotations

import math
import time
from email.utils import parsedate_to_datetime

import httpx
from azure.core.exceptions import ClientAuthenticationError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from cosmos_agentic_retriever.agent.loop import ModelError


class ModelErrorMapper:
    """Keep provider-specific exception details out of the loop and HTTP handler."""

    @classmethod
    def from_exception(cls, error: Exception) -> ModelError:
        """Translate the final failure after SDK retries into a caller-safe error.

        Model authentication is a service configuration problem, not a reason
        to challenge the MCP caller with HTTP 401 and ask them to sign in again.
        """
        status = error.status_code if isinstance(error, APIStatusError) else None
        retry_after = (
            cls._retry_after_seconds(error.response.headers)
            if isinstance(error, APIStatusError) else None
        )
        if isinstance(error, APIStatusError) and cls._is_content_filter(error):
            return cls.content_filter()
        if status == 429:
            return ModelError(
                kind="rate_limit",
                message="The search model's rate limit was reached. Please try again later.",
                status_code=429,
                retry_after_seconds=retry_after,
            )
        if isinstance(error, (APITimeoutError, TimeoutError)) or status in (408, 504):
            return ModelError(
                kind="timeout", message="The search model timed out. Please try again later.",
                status_code=504, retry_after_seconds=retry_after,
            )
        if isinstance(error, ClientAuthenticationError) or status in (401, 403):
            return ModelError(
                kind="authentication",
                message="The search service could not authenticate to the model. Contact the service operator.",
                status_code=502,
            )
        if isinstance(error, APIConnectionError) or (status is not None and status >= 500):
            return ModelError(
                kind="unavailable",
                message="The search model is temporarily unavailable. Please try again later.",
                status_code=503, retry_after_seconds=retry_after,
            )
        if isinstance(error, APIStatusError):
            return ModelError(
                kind="invalid_request",
                message="The model rejected the search request. Contact the service operator.",
                status_code=502,
            )
        if isinstance(error, (TypeError, ValueError)):
            return ModelError(
                kind="invalid_response",
                message="The search model returned an invalid response. Please try again later.",
                status_code=502,
            )
        return ModelError(
            kind="unknown", message="The search model request failed. Please try again later.",
            status_code=502,
        )

    @staticmethod
    def content_filter() -> ModelError:
        """Use the same failure for policy exceptions and successful-but-filtered replies."""
        return ModelError(
            kind="content_filter",
            message="The search model blocked this request under its content policy.",
            status_code=400,
        )

    @staticmethod
    def _retry_after_seconds(headers: httpx.Headers) -> int | None:
        """Normalize provider hints for callers, not for scheduling SDK retries.

        Invalid or past values must not produce misleading or non-finite delays
        in JSON or HTTP headers; fractions round up so callers do not retry early.
        """
        for name in ("retry-after-ms", "x-ms-retry-after-ms", "retry-after"):
            value = headers.get(name)
            if value is None:
                continue
            try:
                delay = float(value) / (1000 if name.endswith("-ms") else 1)
            except ValueError:
                if name != "retry-after":
                    continue
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    continue
            if math.isfinite(delay) and delay >= 0:
                return math.ceil(delay)
        return None

    @staticmethod
    def _is_content_filter(error: APIStatusError) -> bool:
        """Policy blocks can be nested in a 400 response, not a distinct HTTP status."""
        body = error.body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]
        codes = [error.code]
        if isinstance(body, dict):
            codes.append(body.get("code"))
            for key in ("innererror", "inner_error"):
                inner = body.get(key)
                if isinstance(inner, dict):
                    codes.append(inner.get("code"))
        return any(
            isinstance(code, str) and code.casefold() in {
                "content_filter", "content_filter_error", "content_policy_violation",
                "responsibleaipolicyviolation",
            }
            for code in codes
        )