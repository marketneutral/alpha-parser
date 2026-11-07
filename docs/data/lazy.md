# Large Datasets: LazyData

`LazyData` holds a loader per field instead of the field itself, and loads a field the first time an expression reads
it. An expression that uses only `close` never loads `volume` or `earnings`.

```python
from qex import qex, LazyData

data = LazyData({
    'close':    lambda: pd.read_parquet('data/close.parquet'),
    'volume':   lambda: pd.read_parquet('data/volume.parquet'),
    'earnings': lambda: pd.read_parquet('data/earnings.parquet'),
})

result = qex("rank(returns(20))").evaluate(data)    # loads close only
```

A loaded field is kept, so it is read from disk once. Values can be loaders or frames already in memory, mixed.

## Describing the fields

`descriptions` documents each field where the data is defined:

```python
data = LazyData(
    data={
        'close':    lambda: pd.read_parquet('close.parquet'),
        'earnings': lambda: pd.read_parquet('earnings.parquet'),
    },
    descriptions={
        'close':    'Daily close, adjusted for splits and dividends',
        'earnings': 'Quarterly EPS surprise, (actual - estimate) / |estimate|, on the announcement date',
    },
)

print(data.describe())            # every field and its description
print(data.describe('earnings'))  # one field
data.descriptions                 # the dict
```

::: qex.data.LazyData
