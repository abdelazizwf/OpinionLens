from datetime import datetime


def get_timestamp() -> str:
    time = datetime.now().replace(microsecond=0).isoformat()
    replacements = str.maketrans("", "", "T:-")
    time = time.translate(replacements)
    return time
