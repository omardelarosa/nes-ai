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

## Usage

To run any with human keyboard control agent, run:

```
python name_of_agent.py --mode human --rom roms/rom.nes
```

To use the random sample agent

```
python generic_agent.py --mode random --rom roms/rom.nes
```

## Fixing ROMs

Some ROMs require header corrections. Please see [tools/](tools/) directory for instructions on fixing your ROM files.

## Contribution

1. Create a feature branch
2. Complete changes
3. Open a PR against `master` branch
4. Merge PR.
