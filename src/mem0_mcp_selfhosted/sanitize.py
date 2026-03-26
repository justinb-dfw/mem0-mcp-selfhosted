"""Secret and credential sanitizer for mem0 memory content.

Scrubs known secret patterns from text before it is stored in Qdrant.
Applied to all add_memory and update_memory calls.

Patterns are intentionally conservative — we'd rather redact a false
positive than persist a real credential.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern catalog — (label, compiled_regex)
# Each pattern replaces the *entire match* with [REDACTED:<label>].
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic API keys
    ("anthropic-key", re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}", re.ASCII)),
    # OpenAI API keys (sk-proj-... and legacy sk-...)
    ("openai-key", re.compile(r"sk-(?:proj-)?[a-zA-Z0-9]{20,}", re.ASCII)),
    # GitHub tokens (personal access, OAuth, server-to-server, refresh, install)
    ("github-token", re.compile(r"gh[pors]_[a-zA-Z0-9]{36,}", re.ASCII)),
    # AWS access key IDs
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII)),
    # Generic Bearer tokens in Authorization headers
    ("bearer-token", re.compile(r"(?i)Bearer\s+[a-zA-Z0-9\-._~+/]{20,}=*")),
    # PEM private key blocks
    ("pem-private-key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        re.MULTILINE,
    )),
    # PEM certificate blocks (can contain private keys embedded)
    ("pem-certificate", re.compile(
        r"-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----",
        re.MULTILINE,
    )),
    # Inline password/secret/token/api_key assignments (key=value or key: value)
    # Matches: password=abc123, secret: "abc123", api_key = 'abc123', token=abc123
    ("inline-credential", re.compile(
        r"(?i)(?:password|passwd|secret|api[_\-]?key|auth[_\-]?token|access[_\-]?token|private[_\-]?key)"
        r"\s*[=:]\s*[\"']?([^\s\"',;\n]{8,})[\"']?",
    )),
    # Connection strings with embedded credentials
    # Matches: postgresql://user:password@host, mysql://user:pass@host
    ("connection-string", re.compile(
        r"(?i)(?:postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?|amqp|smtp)://"
        r"[^:@\s]+:[^@\s]+@",
    )),
    # JWT tokens (three base64url segments separated by dots)
    ("jwt-token", re.compile(
        r"eyJ[a-zA-Z0-9\-_]+\.eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+",
        re.ASCII,
    )),
    # Slack tokens
    ("slack-token", re.compile(r"xox[baprs]-[a-zA-Z0-9\-]{10,}", re.ASCII)),
    # Stripe keys
    ("stripe-key", re.compile(r"(?:sk|pk|rk)_(?:live|test)_[a-zA-Z0-9]{24,}", re.ASCII)),
    # SendGrid / Twilio / generic SaaS tokens starting with "SG." or "AC"
    ("sendgrid-key", re.compile(r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}", re.ASCII)),
]


def _redact(text: str) -> tuple[str, int]:
    """Apply all patterns to text, returning (redacted_text, redaction_count)."""
    count = 0
    for label, pattern in _PATTERNS:
        def _replacer(m: re.Match, lbl: str = label) -> str:
            nonlocal count
            count += 1
            return f"[REDACTED:{lbl}]"

        text = pattern.sub(_replacer, text)
    return text, count


def sanitize_text(text: str) -> str:
    """Sanitize a plain text string, logging if any redactions occurred."""
    cleaned, n = _redact(text)
    if n:
        logger.warning("sanitize_text: redacted %d secret pattern(s) before storing memory", n)
    return cleaned


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Sanitize the 'content' field of each message dict in a conversation list."""
    total = 0
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            cleaned, n = _redact(content)
            total += n
            result.append({**msg, "content": cleaned})
        else:
            result.append(msg)
    if total:
        logger.warning("sanitize_messages: redacted %d secret pattern(s) across messages", total)
    return result
