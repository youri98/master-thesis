#!/bin/bash

pip install -r requirements.txt
pip install gym[atari]
pip install autorom[accept-rom-license]

mkdir ROMS
python -m atari_py.import_roms ROMS
touch src/key.txt


echo Done!