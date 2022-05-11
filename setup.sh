#!/bin/bash

pip install -r requirements.txt
python -m atari_py.import_roms ROMS
touch key.txt

echo Done!