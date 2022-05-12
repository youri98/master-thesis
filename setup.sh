#!/bin/bash

pip install -r requirements.txt
mkdir ROMS
python -m atari_py.import_roms ROMS
touch src/key.txt


echo Done!