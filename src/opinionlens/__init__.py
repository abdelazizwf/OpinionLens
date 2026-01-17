import os

import dotenv

dotenv.load_dotenv(".env")

print(f"INFO: ENV={os.environ["ENV"]!r}")
