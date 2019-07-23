To set up:
    pip install gym-retro
    python -m retro.import ./roms
    In powershell> cmd /c mklink "$(& python -c "import gym as _; print(_.__path__[0])")\PokemonRed-GameBoy" .\PokemonRed-GameBoy

To run:
    python trainer.py

Where do environments go?:
    python -c "import gym as _; print(_.__path__)"


To make development easier:
    mklink /d path\to\python\lib\Lib\site-packages\retro\data\stable\PokemonRed-GameBoy path\to\src\PokemonRed-GameBoy

Useful links:

https://gym.openai.com/docs/

https://github.com/openai/retro/tree/develop