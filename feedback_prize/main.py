import os
from pathlib import Path
from utils import is_script_running
from dotenv import load_dotenv

import pandas as pd

load_dotenv()

ON_KAGGLE: bool = is_script_running()
DATA_ROOT = Path(os.path.join(
    '../input', os.environ['PROJECT']) if ON_KAGGLE else os.path.join('/content', os.environ['PROJECT']))

train = pd.read_csv(DATA_ROOT / 'train.csv')
test = pd.read_csv(DATA_ROOT / 'test.csv')
sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')


def main():
    print('Hello world!')


if __name__ == '__main__':
    main()
