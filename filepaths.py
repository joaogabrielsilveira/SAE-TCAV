import os
import sys
from pathlib import Path

def get_env_path(path):
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        from google.colab import drive
        drive.mount('content/drive')
        BASE_DIR = 'content/drive/SAE-TCAV'
    else:
        BASE_DIR = ''

    final_path = os.path.join(BASE_DIR, path)
    folder_path = os.path.dirname(final_path)

    if folder_path:
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    return final_path