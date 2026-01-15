import os
import sys

import httpx


def main():
    files = sys.argv[1:]

    for file in files:
        assert os.path.exists(file)

        f = {'file': open(file, 'rb')}

        url = os.environ["API_SAVED_OBJECTS_URL"]
        resp = httpx.post(
            url=url,
            files=f,
            headers={"X-key": os.environ["ADMIN_API_KEY"]},
        )
        print(resp.json())


if __name__ == "__main__":
    main()
