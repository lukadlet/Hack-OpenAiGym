To set up:
    pip install gym-retro
    In powershell> cmd /c mklink /d "$(& python -c "import retro as _; print(_.__path__[0])")\data\stable\PokemonRed-GameBoy" "$(pwd)\PokemonRed-GameBoy"
    python -m retro.import ./PokemonRed-GameBoy

To run:
    python trainer.py

Useful links:

https://gym.openai.com/docs/

https://github.com/openai/retro/tree/develop