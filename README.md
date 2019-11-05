# Nintendo AI

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Nintendo_gray_logo.svg/200px-Nintendo_gray_logo.svg.png)

General AI Agents, wrappers and tools for Nintendo/Famicom ROMs.

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

To run any agent, run:

```
python name-of-agent.py
```

For example:

```
python megaman_agent.py
```

## Fixing ROMs

Some ROMs require header corrections. Please see [tools/](tools/) directory for instructions on fixing your ROM files.

## Contribution

1. Create a feature branch
2. Complete changes
3. Open a PR against `master` branch
4. Merge PR.
