import os
import sys
from pathlib import Path

def get_env_path(path):
    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False

    if IN_COLAB:
        BASE_DIR = '/content/drive/MyDrive/IC/SAE-TCAV'
    else:
        BASE_DIR = ''

    final_path = os.path.join(BASE_DIR, path)
    folder_path = os.path.dirname(final_path)

    if folder_path:
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    return final_path