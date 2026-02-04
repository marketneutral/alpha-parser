# Installation

## Requirements

- Python 3.9+
- numpy, pandas, scipy

## Install from PyPI

```bash
pip install qex
```

## Install from Git

```bash
# Basic
pip install git+https://github.com/marketneutral/alpha-parser.git

# With all optional dependencies
pip install "qex[all] @ git+https://github.com/marketneutral/alpha-parser.git"
```

## Install for Development

```bash
git clone https://github.com/marketneutral/alpha-parser
cd alpha-parser
pip install -e ".[all]"
```

## Optional Dependencies

| Group | Packages | Use Case |
|-------|----------|----------|
| `dev` | pytest, pyarrow | Running tests |
| `data` | requests, python-dotenv, pyarrow | FMP data fetcher |
| `risk` | statsmodels | Factor risk model |
| `all` | Everything above | Full installation |

Install specific groups:

```bash
pip install "qex[dev]"      # Just testing
pip install "qex[risk]"     # With risk model
pip install "qex[all]"      # Everything
```

## Verify Installation

```python
from qex import qex

signal = qex("rank(returns(20))")
print(signal)  # Should print: Rank(Returns(20))
```
