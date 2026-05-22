import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text.

    This function removes extra spaces, repeated blank lines,
    and unnecessary invisible characters.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove too many spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before new lines
    text = re.sub(r" +\n", "\n", text)

    return text.strip()