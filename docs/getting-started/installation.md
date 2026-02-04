# Installation

## Requirements

- Python 3.9+
- numpy, pandas, scipy

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

Install specific groups (from cloned repo):

```bash
pip install -e ".[dev]"      # Just testing
pip install -e ".[risk]"     # With risk model
pip install -e ".[all]"      # Everything
```

## Verify Installation

```python
from qex import qex

signal = qex("rank(returns(20))")
print(signal)  # Should print: Rank(Returns(20))
```
