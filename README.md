# Nintendo AI

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Nintendo_gray_logo.svg/200px-Nintendo_gray_logo.svg.png)

Generic Open AI Gym Agents, wrappers and tools for Nintendo/Famicom ROMs.

BYOR (Bring your own roms)

## Installation Requirements

Both of these libraries must be installed.

[OpenAI Gym](https://github.com/openai/gym)

[nes-py](https://github.com/Kautenja/nes-py)

It's recommended that they are cloned as repositories and installed from git repos using:

```
python -e .
```

This allows for extensibility.

### Install script
```
mkdir nes-ai-root
cd nes-ai-root

git clone https://github.com/openai/gym.git
cd gym
pip3 install -e .
cd ..

git clone https://github.com/Kautenja/nes-py.git
cd nes-py
pip3 install -e .
cd ..

git clone https://github.com/omardelarosa/nes-ai.git
cd nes-ai
pip3 install -r requirements.txt
```

## Usage

To run any with human keyboard control agent, run:

```
python . -a RandomAgent --mode human --rom roms/rom.nes
```

To use the random sample agent

```
python . -a RandomAgent--mode random --rom roms/rom.nes
```

## Adding Agents

To add an agent:

1. create a file in the `agents` directory using the naming convention `AgentName.py`.

2. The file must implement and export a class matching this signature:

```python
class AgentName():
    def __init__(self, args)
        # do stuff here
```

3. Register your class in `agents/__init__.py` as follows:

```python
from . import AgentName

__all__ = [
    'AgentName'
]
```

## Fixing ROMs

Some ROMs require header corrections. Please see [tools/](tools/) directory for instructions on fixing your ROM files.

## Contribution

1. Create a feature branch
2. Complete changes
3. Open a PR against `master` branch
4. Merge PR.
