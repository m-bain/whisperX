import os
from typing import Optional

HF_TOKEN_ENV_VAR = "HF_TOKEN"


class HFTokenError(ValueError):
    """Raised when the Hugging Face token cannot be resolved or is invalid.

    Messages never include the token value, so they are safe to print.
    """


def _validate(token: str, source: str) -> str:
    # Reject embedded whitespace/control characters. Such a value would be
    # sent as an HTTP header, and the resulting error from `requests` would
    # print the full header value, leaking the token.
    if any(ch.isspace() or not ch.isprintable() for ch in token):
        raise HFTokenError(
            f"Hugging Face token from {source} contains whitespace or control characters"
        )
    return token


def _read_token_file(hf_token_file: str) -> str:
    path = os.path.expanduser(hf_token_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            token = f.read().strip()
    except FileNotFoundError:
        raise HFTokenError(f"Hugging Face token file not found: {hf_token_file}") from None
    except UnicodeDecodeError:
        raise HFTokenError(f"Hugging Face token file is not valid UTF-8: {hf_token_file}") from None
    except OSError as e:
        reason = e.strerror or type(e).__name__
        raise HFTokenError(f"Cannot read Hugging Face token file {hf_token_file}: {reason}") from None
    if not token:
        raise HFTokenError(f"Hugging Face token file is empty: {hf_token_file}")
    return _validate(token, "--hf_token_file")


def resolve_hf_token(
    hf_token: Optional[str] = None,
    hf_token_file: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the Hugging Face access token from the available sources.

    ``hf_token`` and ``hf_token_file`` are mutually exclusive. The token is
    taken from, in order:
        1. ``hf_token`` or ``hf_token_file`` (whichever one was given)
        2. The ``HF_TOKEN`` environment variable
        3. ``None``, which leaves authentication to huggingface_hub (for
           example a token saved with ``hf auth login``)

    Passing secrets on the command line exposes them to other users via the
    process list, so the file or environment variable is preferred.

    Raises:
        HFTokenError: If both options are given, the token file cannot be
            read or is empty, or the token contains whitespace or control
            characters. The message never contains the token.
    """
    if hf_token is not None and hf_token_file is not None:
        raise HFTokenError("Use only one of --hf_token and --hf_token_file, not both.")

    if hf_token is not None:
        return _validate(hf_token.strip(), "--hf_token")

    if hf_token_file is not None:
        return _read_token_file(hf_token_file)

    env_token = os.environ.get(HF_TOKEN_ENV_VAR, "").strip()
    if env_token:
        return _validate(env_token, f"the {HF_TOKEN_ENV_VAR} environment variable")

    return None
