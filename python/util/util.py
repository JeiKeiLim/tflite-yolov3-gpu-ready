import os
import requests
from tqdm import tqdm


def download_wget(url, target_path, exist_check=True):
    r = requests.get(url, stream=True)

    if exist_check and os.path.exists(target_path):
        return False
    try:
        with tqdm.wrapattr(open(target_path, "wb"), "write",
                           miniters=1, desc=url.split('/')[-1],
                           total=int(r.headers.get('content-length', 0))) as fout:
            for chunk in r.iter_content(chunk_size=4096):
                fout.write(chunk)
    except:
        return False

    return True
