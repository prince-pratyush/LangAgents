import sys
import traceback
from typing import Optional


class ResearchAnalystException(Exception):
    """
    Base exception for the Research & Analyst AI/LLM pipeline.

    Wraps any upstream exception and enriches it with:
    - The exact file and line number where the failure originated
    - The full traceback string for downstream logging
    - An optional free-form context dict (user_id, model_name, db_name, …)
    - A compact, single-line repr suitable for structured log fields

    Typical usage
    -------------
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise ResearchAnalystException(
            "LLM invocation failed",
            cause=e,
            context={"model": "gpt-4o", "user_id": "u_42"},
        ) from e
    """

    def __init__(
        self,
        message: str,
        cause: Optional[BaseException] = None,
        context: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        message:
            A short, human-readable description of what went wrong.
        cause:
            The original exception being wrapped. When omitted the active
            exception from sys.exc_info() is used automatically, so the class
            works equally well inside and outside an except block.
        context:
            Arbitrary key-value metadata relevant to this error
            (e.g. ``{"user_id": "u_42", "model": "gpt-4o", "db": "chroma"}``).
        """
        self.message = str(message)
        self.context: dict = context or {}

        # ── Resolve the source exception ──────────────────────────────────
        if cause is not None:
            exc_type, exc_value, exc_tb = type(cause), cause, cause.__traceback__
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # ── Walk to the deepest frame (actual failure site) ───────────────
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name: str = (
            last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        )
        self.lineno: int = last_tb.tb_lineno if last_tb else -1

        # ── Capture the full formatted traceback ──────────────────────────
        if exc_type is not None and exc_tb is not None:
            self.traceback_str: str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        super().__init__(self.message)

    # ── Formatting helpers ────────────────────────────────────────────────

    def _context_str(self) -> str:
        """Render context dict as a compact inline string."""
        if not self.context:
            return ""
        pairs = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f" | context=[{pairs}]"

    def compact(self) -> str:
        """
        Single-line summary — safe to pass directly to a structured logger.

        Example output
        --------------
        [pipeline/llm_chain.py:84] LLM invocation failed | context=[model='gpt-4o', user_id='u_42']
        """
        return (
            f"[{self.file_name}:{self.lineno}] "
            f"{self.message}"
            f"{self._context_str()}"
        )

    def __str__(self) -> str:
        """
        Multi-line string that includes the full traceback when available.
        Used automatically by Python when the exception is printed or logged.
        """
        if self.traceback_str:
            return f"{self.compact()}\nTraceback:\n{self.traceback_str.rstrip()}"
        return self.compact()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"file={self.file_name!r}, "
            f"line={self.lineno}, "
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )

    def to_dict(self) -> dict:
        """
        Serialise the exception to a plain dict — ready to pass directly to
        a structlog logger or any JSON serialiser.

        Example
        -------
        logger.error("pipeline_error", **exc.to_dict())
        """
        return {
            "error": self.message,
            "file": self.file_name,
            "line": self.lineno,
            "traceback": self.traceback_str.strip() or None,
            **self.context,
        }


# ── Domain-specific subclasses ────────────────────────────────────────────────
# Subclass for every distinct failure domain so callers can catch selectively.
#
#   except LLMException:          # only LLM errors
#   except ResearchAnalystException:  # any pipeline error

class LLMException(ResearchAnalystException):
    """Raised when an LLM API call fails (timeout, rate-limit, bad response)."""


class EmbeddingException(ResearchAnalystException):
    """Raised when an embedding model call fails."""


class VectorDBException(ResearchAnalystException):
    """Raised when a vector-store read/write operation fails."""


class ConfigException(ResearchAnalystException):
    """Raised when a required configuration key is missing or invalid."""


class ValidationException(ResearchAnalystException):
    """Raised when input or output schema validation fails."""
