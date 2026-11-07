# Calendar

The date as a number, the same for every stock on a day.

### day_of_week

<p class="sig"><code>day_of_week()</code></p>

0 (Monday) to 6 (Sunday). The toy data runs Tuesday Jan 02 to Tuesday Jan 09, over a weekend.

```toy
day_of_week()
```

```qex
where(day_of_week() == 4, rank(-returns(5)), None)         # trade on Fridays only
```

### day_of_month

<p class="sig"><code>day_of_month()</code></p>

1 to 31.

```qex
day_of_month() <= 5                                         # the first days of the month
```

### month_of_year

<p class="sig"><code>month_of_year()</code></p>

1 to 12.

```qex
where(month_of_year() == 1, 1, 0)                           # a January indicator
month_of_year() != delay(month_of_year(), 1)                # the first trading day of each month
```
