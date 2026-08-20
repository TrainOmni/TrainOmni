import time
import traceback

from huggingface_hub import snapshot_download


while True:
    try:
        snapshot_download(
            repo_id="openbmb/MiniCPM5-1B",
            local_dir=r"D:\Models\LLM\MiniCPM5-1B",
            cache_dir=r"D:\Models\_cache\huggingface\hub",
            max_workers=1,
        )
        break
    except Exception:
        traceback.print_exc()
        print("Download interrupted; retrying from the saved checkpoint in 10 seconds.", flush=True)
        time.sleep(10)
