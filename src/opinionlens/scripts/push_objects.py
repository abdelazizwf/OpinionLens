import os
import sys

import httpx

from opinionlens.common.settings import get_settings

settings = get_settings()


def main():
    files = sys.argv[1:]

    for file in files:
        assert os.path.exists(file)

        f = {'file': open(file, 'rb')}

        if type(settings.api.admin_key) is not str:
            key = settings.api.admin_key.get_secret_value()
        else:
            key = settings.api.admin_key

        url = settings.api.object_upload_url
        resp = httpx.post(
            url=url,
            files=f,
            headers={"X-key": key},
        )
        print(resp.json())


if __name__ == "__main__":
    main()
