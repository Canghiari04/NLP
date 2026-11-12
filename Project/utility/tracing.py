def banner(message: str, char: str = "-") -> str:
    return f"{char * len(message)}\n{message}\n{char * len(message)}"