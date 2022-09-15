import gzip
import base64
import os
from pathlib import Path
from typing import Any, Dict


# this is base64 encoded source code
file_data: Dict = {file_data}  # type:ignore


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python feedback_prize/main.py')
