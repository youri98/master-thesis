
pip install -r requirements.txt

python -m atari_py.import_roms ROMS
# touch src/key.txt

ale-import-roms --import-from-pkg atari_py.atari_roms

echo Done!