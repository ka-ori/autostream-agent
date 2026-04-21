def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock lead capture API. Prints confirmation to stdout."""
    msg = f"Lead captured successfully: {name}, {email}, {platform}"
    print(f"\n[LEAD CAPTURE] {msg}")
    return msg
