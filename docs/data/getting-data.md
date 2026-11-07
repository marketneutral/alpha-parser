# Getting Data

qex doesn't ship data or depend on a vendor: anything you can put into the [data format](format.md) works. Two
starting points come with the repository.

## Synthetic data

The [Quick Start](../python-examples/quickstart.md) and the [complete dataset example](format.md#example-complete-dataset)
build random prices, volumes, sectors and earnings in a few lines: enough to try every operator.

## Financial Modeling Prep

`data/fetch_fmp.py` downloads daily prices, volumes and company profiles from
[Financial Modeling Prep](https://financialmodelingprep.com/). It needs an FMP API key with access to FMP's
`api/v3` endpoints; without one, FMP answers 403 and the script stops.

```bash
echo "FMP_API_KEY=your-key" > data/.env
pip install -e ".[data]"

python data/fetch_fmp.py                                    # S&P 500, last 5 years
python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN      # chosen tickers
python data/fetch_fmp.py --start 2020-01-01 --end 2024-01-01
```

It writes one parquet file per field to `data/fmp/`: `open`, `high`, `low`, `close`, `volume`, `sector`, `industry`
and `profiles`. Load them lazily:

```python
from qex import qex, LazyData

data = LazyData({
    'close':  lambda: pd.read_parquet('data/fmp/close.parquet'),
    'volume': lambda: pd.read_parquet('data/fmp/volume.parquet'),
    'sector': lambda: pd.read_parquet('data/fmp/sector.parquet'),
})
```

FMP's sector and industry are today's, applied to every past day: see the warning in
[Labels and Groups](labels.md). The integration tests use this data and skip themselves when it isn't there.

## Limits: restated data

qex assumes each value is final: one value per date and stock. It does not support **point-in-time (bi-temporal)
data**, where a past value is later revised.

```
Jan 8:   a company reports earnings of $1.00
Jan 31:  the figure is restated to $3.00

ts_mean(field('earnings'), 20) on Jan 15 should see $1.00 (what was known then),
and on Feb 5 should see $3.00. One cell, (Jan 8, the company), can hold only one of them.
```

Prices are rarely restated; fundamentals often are. The practical choices:

1. **Use data that isn't restated**, or the first-reported values (an "as first reported" or point-in-time feed).
2. **Accept the bias** for research, knowing that restated fundamentals know their own future.
3. **Build point-in-time tables outside qex**, one set per evaluation period, in your data pipeline.
