# Current results and interpretations

Updated August 30, 2026. These results use the corrected $[T-60,T)$ settlement implementation and underlying market data available through August 26, 2026. All reported P&L is settlement P&L with zero modeled maker fees, consistent with the historical fee configuration.

## Probability backtest

There were 3,735 markets discovered, of which 3,483 had exchange-recorded strikes and settlements. Approximately 2,850 markets were available for out-of-sample testing at each horizon. The exact count ranged from 2,847 to 2,853.

### Structural and Residual-Calibrated Log Loss across Forecast Horizons

| Seconds to Expiration | Residual-calibrated log loss | Uncalibrated structural log loss | Calibration improvement | 95% day-block interval for log-loss improvement |
| ---: | ---: | ---: | ---: | ---: |
| 30 seconds | 0.03556 | 0.03749 | 0.00193 | -0.00018 to 0.00403 |
| 60 seconds | 0.11609 | 0.11990 | 0.00381 | 0.00047 to 0.00730 |
| 120 seconds | 0.22169 | 0.22334 | 0.00165 | -0.00110 to 0.00469 |
| 300 seconds | 0.37186 | 0.37378 | 0.00192 | -0.00030 to 0.00425 |
| 600 seconds | 0.55791 | 0.55777 | -0.00014 | -0.00131 to 0.00112 |

Log loss measures the accuracy and calibration of predicted probabilities, with lower values indicating better forecasts. Calibration improvement is calculated as uncalibrated structural log loss minus residual-calibrated log loss, so positive values indicate improvement. The largest improvement occurs at 60 seconds, where the 95% day-block interval for log-loss improvement does not include zero. At the other horizons, the observed differences may be due to sampling variation because their 95% intervals include zero. These results support residual calibration as a short-horizon improvement but do not indicate effectiveness at all horizons.

## Last-minute maker replay

The maker replay contains 2,858 markets with exchange-recorded strikes and settlements across 35 UTC days, 25 of which had at least one fill. Quoting starts at $T-60$ and stops at $T-15$, using a 250 ms latency assumption.

This remains a counterfactual replay: simulated orders could have changed other participants' behavior, and actual private queue position and order acknowledgements were unavailable.

### Last-minute maker results by size

| Fill model | Minimum model-to-quote distance | Size | Order size | Inventory limit | Quoted markets | Markets with fills | Fill events | Filled quantity | P&L | P&L/contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | 1x | 1 | 10 | 1,098 | 557 | 4,616 | 4,154.31 | \$107.19 | 2.58 cents |
| Trade-through | 1 cent | 10x | 10 | 100 | 1,098 | 557 | 5,930 | 37,415.11 | \$909.70 | 2.43 cents |
| Trade-through | 2 cents | 1x | 1 | 10 | 866 | 447 | 3,487 | 3,158.33 | \$98.34 | 3.11 cents |
| Trade-through | 2 cents | 10x | 10 | 100 | 866 | 447 | 4,380 | 28,573.30 | \$833.35 | 2.92 cents |
| Queue-aware | 1 cent | 1x | 1 | 10 | 1,098 | 571 | 5,120 | 4,490.45 | \$118.84 | 2.65 cents |
| Queue-aware | 1 cent | 10x | 10 | 100 | 1,098 | 571 | 6,650 | 40,608.47 | \$1,029.01 | 2.53 cents |
| Queue-aware | 2 cents | 1x | 1 | 10 | 866 | 452 | 3,871 | 3,421.39 | \$112.85 | 3.30 cents |
| Queue-aware | 2 cents | 10x | 10 | 100 | 866 | 452 | 4,920 | 31,036.53 | \$989.68 | 3.19 cents |

### 10x scaling relative to 1x

| Fill model | Minimum model-to-quote distance | Filled-volume multiple | P&L multiple | Change in P&L/contract |
| --- | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | 9.01x | 8.49x | -5.8% |
| Trade-through | 2 cents | 9.05x | 8.47x | -6.3% |
| Queue-aware | 1 cent | 9.04x | 8.66x | -4.2% |
| Queue-aware | 2 cents | 9.07x | 8.77x | -3.3% |

Quotes use the inventory-adjusted reservation probability defined in the [backtesting methodology](backtesting.md). The 1x scenarios use one-contract orders and a 10-contract inventory limit. The 10x scenarios use ten-contract orders and a 100-contract limit. Inventory skew is scaled so both settings permit the same maximum two-cent adjustment.

The two-cent setting generated fewer quotes, fills, and contracts than the one-cent setting, but more P&L per contract. The lower total P&L reflects a tradeoff between volume and profit margin. The 10x tests produced approximately nine times the 1x filled volume, while P&L per contract decreased by 3.3% to 6.3%.

## Alternative probability calculations

Two alternatives were tested using the same markets and execution assumptions. The Monte Carlo calculation used 8,192 deterministic Sobol paths to evaluate the correlated GBM sum directly, followed by probability integral transform (PIT) residual calibration using only prior-day data. Residual calibration modifies the structural forecast by replacing its standard-normal residual distribution with the prior-days-only empirical residual distribution. The pure empirical model instead replaces the structural forecast entirely with the prior-days-only distribution of the realized remaining-sample average relative to decision-time spot. It does not use GBM or current volatility. A separate distribution was maintained for every second from 15 through 60 seconds before expiration.

### Probability-model economic comparison

Each model column reports total P&L followed by P&L per contract. Difference intervals use 10,000 paired bootstrap resamples of the 35 UTC test days.

| Fill model | Minimum model-to-quote distance | Size | Moment-matched | Monte Carlo | MC minus moment-matched P&L (95% paired UTC-day interval) | Pure empirical | Pure empirical minus moment-matched P&L (95% paired UTC-day interval) | Moment-matched significantly better than pure empirical at 5%? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | 1x | \$107.19 / 2.58 cents | \$105.32 / 2.52 cents | -\$1.87 (-\$5.67 to \$1.65) | \$58.41 / 1.28 cents | -\$48.79 (-\$107.31 to \$8.56) | No |
| Trade-through | 1 cent | 10x | \$909.70 / 2.43 cents | \$913.41 / 2.43 cents | \$3.71 (-\$41.11 to \$52.38) | \$508.02 / 1.23 cents | -\$401.68 (-\$924.00 to \$110.47) | No |
| Trade-through | 2 cents | 1x | \$98.34 / 3.11 cents | \$98.88 / 3.13 cents | \$0.54 (-\$3.24 to \$4.73) | \$64.31 / 1.75 cents | -\$34.03 (-\$84.55 to \$17.57) | No |
| Trade-through | 2 cents | 10x | \$833.35 / 2.92 cents | \$845.76 / 2.95 cents | \$12.41 (-\$19.43 to \$46.28) | \$537.62 / 1.61 cents | -\$295.72 (-\$777.78 to \$175.42) | No |
| Queue-aware | 1 cent | 1x | \$118.84 / 2.65 cents | \$118.95 / 2.62 cents | \$0.11 (-\$4.08 to \$4.03) | \$72.05 / 1.46 cents | -\$46.78 (-\$104.71 to \$11.68) | No |
| Queue-aware | 1 cent | 10x | \$1,029.01 / 2.53 cents | \$1,040.95 / 2.55 cents | \$11.94 (-\$32.95 to \$57.95) | \$630.76 / 1.41 cents | -\$398.25 (-\$929.38 to \$139.06) | No |
| Queue-aware | 2 cents | 1x | \$112.85 / 3.30 cents | \$112.69 / 3.29 cents | -\$0.16 (-\$3.25 to \$2.62) | \$77.52 / 1.96 cents | -\$35.33 (-\$88.58 to \$18.60) | No |
| Queue-aware | 2 cents | 10x | \$989.68 / 3.19 cents | \$995.21 / 3.20 cents | \$5.53 (-\$20.21 to \$29.87) | \$657.37 / 1.83 cents | -\$332.31 (-\$824.76 to \$169.26) | No |

### Probability-model execution comparison

Each model column reports markets with fills followed by filled quantity. The moment-matched model quoted 1,098 markets at one cent and 866 at two cents. The corresponding counts were 1,093 and 863 for Monte Carlo and 1,049 and 883 for pure empirical.

| Fill model | Minimum model-to-quote distance | Size | Moment-matched | Monte Carlo | Pure empirical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | 1x | 557 / 4,154.31 | 565 / 4,183.54 | 647 / 4,572.28 |
| Trade-through | 1 cent | 10x | 557 / 37,415.11 | 565 / 37,656.12 | 647 / 41,304.35 |
| Trade-through | 2 cents | 1x | 447 / 3,158.33 | 449 / 3,164.58 | 550 / 3,686.18 |
| Trade-through | 2 cents | 10x | 447 / 28,573.30 | 449 / 28,647.07 | 550 / 33,469.34 |
| Queue-aware | 1 cent | 1x | 571 / 4,490.45 | 580 / 4,535.01 | 669 / 4,932.16 |
| Queue-aware | 1 cent | 10x | 571 / 40,608.47 | 580 / 40,793.57 | 669 / 44,797.12 |
| Queue-aware | 2 cents | 1x | 452 / 3,421.39 | 453 / 3,421.63 | 565 / 3,946.74 |
| Queue-aware | 2 cents | 10x | 452 / 31,036.53 | 453 / 31,092.56 | 565 / 36,014.05 |

Monte Carlo changed P&L only marginally while being much more computationally intensive. Furthermore, all eight 95% paired UTC-day intervals for the Monte Carlo-minus-moment-matched difference included zero. This indicates that the lognormal approximation is adequate and that Monte Carlo does not provide an established improvement over moment matching.

The pure empirical model produced only about 55% to 69% of moment-matched P&L and less P&L per contract in all eight variants. However, all eight paired UTC-day intervals for the pure-empirical-minus-moment-matched difference included zero. The observed direction favors moment matching, but this sample does not establish its statistical superiority at the 5% level.

## Full-market replay

The residual-calibrated moment-matching model was also tested continuously from market open at $T-900$ through $T-15$.

### Last-minute versus full-market performance, volume, drawdown, and uncertainty

The 95% intervals are standalone total-P&L intervals from 10,000 UTC-day block bootstrap resamples. Maximum drawdown is calculated from cumulative settlement P&L.

#### 1x scenarios

| Fill model | Edge | Window | Filled quantity | P&L | P&L/contract | Maximum drawdown | Total-P&L 95% interval |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | Last-60 | 4,154.31 | \$107.19 | 2.580 cents | -\$19.68 | \$21.88 to \$199.26 |
| Trade-through | 1 cent | Full market | 77,738.62 | -\$44.36 | -0.057 cents | -\$165.70 | -\$277.53 to \$176.57 |
| Trade-through | 2 cents | Last-60 | 3,158.33 | \$98.34 | 3.114 cents | -\$21.46 | \$12.79 to \$189.25 |
| Trade-through | 2 cents | Full market | 55,891.73 | \$91.84 | 0.164 cents | -\$139.46 | -\$136.76 to \$318.82 |
| Queue-aware | 1 cent | Last-60 | 4,490.45 | \$118.84 | 2.646 cents | -\$20.42 | \$28.27 to \$211.19 |
| Queue-aware | 1 cent | Full market | 89,697.09 | \$165.82 | 0.185 cents | -\$119.83 | -\$94.03 to \$417.85 |
| Queue-aware | 2 cents | Last-60 | 3,421.39 | \$112.85 | 3.298 cents | -\$20.83 | \$23.87 to \$206.11 |
| Queue-aware | 2 cents | Full market | 63,987.02 | \$270.33 | 0.422 cents | -\$99.83 | \$27.70 to \$516.31 |

#### 10x scenarios

| Fill model | Edge | Window | Filled quantity | P&L | P&L/contract | Maximum drawdown | Total-P&L 95% interval |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Trade-through | 1 cent | Last-60 | 37,415.11 | \$909.70 | 2.431 cents | -\$197.18 | \$123.89 to \$1,747.69 |
| Trade-through | 1 cent | Full market | 679,085.40 | -\$326.14 | -0.048 cents | -\$1,577.25 | -\$2,669.67 to \$1,935.17 |
| Trade-through | 2 cents | Last-60 | 28,573.30 | \$833.35 | 2.917 cents | -\$225.19 | \$32.52 to \$1,672.06 |
| Trade-through | 2 cents | Full market | 495,557.26 | \$834.72 | 0.168 cents | -\$1,384.27 | -\$1,448.48 to \$3,071.15 |
| Queue-aware | 1 cent | Last-60 | 40,608.47 | \$1,029.01 | 2.534 cents | -\$201.60 | \$178.43 to \$1,912.64 |
| Queue-aware | 1 cent | Full market | 780,203.91 | \$1,566.92 | 0.201 cents | -\$1,122.87 | -\$907.44 to \$4,057.58 |
| Queue-aware | 2 cents | Last-60 | 31,036.53 | \$989.68 | 3.189 cents | -\$220.15 | \$128.07 to \$1,855.10 |
| Queue-aware | 2 cents | Full market | 564,638.57 | \$2,421.47 | 0.429 cents | -\$1,025.99 | -\$59.06 to \$4,902.62 |

Full-market quoting greatly increased volume, but profit per contract fell and drawdowns increased. Only the full-market queue-aware 2-cent 1x result had a positive 95% total P&L interval.

### 1x settlement P&L by fill horizon

| Time to expiration | Trade-through 1 cent | Trade-through 2 cents | Queue-aware 1 cent | Queue-aware 2 cents |
| --- | ---: | ---: | ---: | ---: |
| 900-600 seconds | -\$14.66 | \$54.20 | \$75.44 | \$117.46 |
| 600-300 seconds | \$63.09 | \$52.71 | \$143.49 | \$128.63 |
| 300-120 seconds | -\$76.56 | -\$34.07 | -\$72.34 | -\$28.72 |
| 120-60 seconds | -\$72.28 | -\$50.61 | -\$49.39 | -\$35.50 |
| 60-30 seconds | \$16.05 | \$32.96 | \$27.56 | \$48.11 |
| 30-15 seconds | \$37.77 | \$34.51 | \$38.72 | \$38.01 |
| 15-0 seconds, stale orders only | \$2.23 | \$2.15 | \$2.35 | \$2.35 |

Consistent weakness was found in the 300-60 second bands. Both the 600-300 and the 60-15 windows appear promising. The 900-600 second results are less reliable because forecasts earlier than 600 seconds reuse the 600-second residual distribution.

The horizon table comes from one continuous strategy, so inventory carries between bands. Separate band-only tests are required before treating the profitable windows as an improvement.

## Out-of-sample period

The 860 markets added after the original August 14 replay form a temporal out-of-sample period for the unchanged strategy specifications. They were profitable in every 10x last-minute scenario. The intervals use 10,000 UTC-day block bootstrap resamples of the 11 added-period days:

### Added-period 10x last-minute performance

| Scenario | Added-period P&L | P&L/contract | Total-P&L 95% interval |
| --- | ---: | ---: | ---: |
| Trade-through, 1-cent edge | \$282.20 | 1.49 cents | -\$192.07 to \$701.09 |
| Trade-through, 2-cent edge | \$218.63 | 1.53 cents | -\$309.05 to \$679.97 |
| Queue-aware, 1-cent edge | \$298.17 | 1.47 cents | -\$177.99 to \$718.49 |
| Queue-aware, 2-cent edge | \$266.03 | 1.75 cents | -\$262.23 to \$716.54 |

The new period supports persistence of the sign of the result, but all four intervals include zero. Its smaller per-contract P&L also suggests that the original effect size should not be extrapolated unchanged.

## Interpretation

The evidence currently supports the following conclusions:

- Modeling the market's settlement rule produces informative short-horizon probabilities, especially near expiration.
- Moment matching and Monte Carlo were economically approximately equal. Due to Monte Carlo's computational intensity, there is no clear benefit to using it.
- The structural model, which accounts for current volatility, produced higher P&L than the purely empirical model in all eight variants, though the paired differences were not significant at the 5 percent level. The results favor retaining current-volatility conditioning but do not show that volatility alone caused the difference.
- Historical markets seem to support scaling to higher inventory and order sizes, though 10x order sizing resulted in about 9x filled volume and 8.47-8.77x total P&L, suggesting diminishing returns to flatly increasing order size.
- Full-market quoting dilutes the edge, although it increases volume, and most edge seems to be concentrated in certain time bands.

These results do not establish that live orders will receive the same fills, that the best historical configuration is the true best configuration, that the profitable time bands will remain profitable when traded independently, or that future P&L will match the historical totals.

## Next steps

The queue-aware model with a two-cent minimum model-to-quote distance and 10x order sizing should be frozen and tested prospectively for 45-60 active days. Alternative market windows and dynamic order-sizing models should also be tested as exploratory strategies because there seem to be diminishing returns to flatly increasing order size.
