import pytest
from agentapi.core.redaction import DEFAULT_PATTERNS, redact, redact_exception, REDACTED


class TestDefaultPatterns:
    def test_openai_key_redacted(self):
        text = "Auth failed for sk-abcdefghij1234567890XYZ"
        assert "sk-" not in redact(text)
        assert REDACTED in redact(text)

    def test_groq_key_redacted(self):
        text = "Invalid token: gsk_abcdefghij1234567890"
        assert "gsk_" not in redact(text)

    def test_anthropic_key_redacted(self):
        text = "Bad key sk-ant-abcdefghij1234567890XYZ"
        assert "sk-ant-" not in redact(text)

    def test_google_key_redacted(self):
        text = "AIzaSyABCDEFGHIJ1234567890abcdefghij12345 invalid"
        assert "AIza" not in redact(text)

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def"
        assert "Bearer eyJ" not in redact(text)

    def test_credential_url_redacted(self):
        text = "Connection failed: https://admin:secret@db.example.com/foo"
        result = redact(text)
        assert "admin:secret" not in result
        assert "@db.example.com" not in result

    def test_no_match_returns_original(self):
        text = "Generic error message with no secrets"
        assert redact(text) == text

    def test_multiple_matches_all_redacted(self):
        text = "Key sk-aaaaaaaaaaaaaaaaaaaa and key sk-bbbbbbbbbbbbbbbbbbbb"
        result = redact(text)
        assert result.count(REDACTED) == 2
        assert "sk-a" not in result and "sk-b" not in result


class TestCustomPatterns:
    def test_custom_pattern_applied(self):
        text = "internal-secret-XYZ123 leaked"
        result = redact(text, patterns=[r"internal-secret-[A-Z0-9]+"])
        assert "internal-secret-" not in result

    def test_custom_replacement_string(self):
        text = "sk-abcdefghij1234567890XYZ"
        result = redact(text, replacement="***")
        assert result == "***"


class TestRedactException:
    def test_redacts_exception_message(self):
        try:
            raise ValueError("API call failed with key sk-abcdefghij1234567890XYZ")
        except ValueError as exc:
            result = redact_exception(exc)
            assert "sk-" not in result
            assert REDACTED in result

    def test_preserves_non_sensitive_content(self):
        exc = RuntimeError("Connection timeout after 30s")
        result = redact_exception(exc)
        assert result == "Connection timeout after 30s"