# Backtesting and Execution

## Two separate backtests

The repository evaluates two questions independently.

- Are the fair probabilities accurate and calibrated?
- Could resting orders based on those probabilities have been executed profitably under the stated assumptions?

## Settlement labels and data quality

KXDOGE15M settles from the 60 one-second RTI observations in $[T-60,T)$, as specified by the contract. The database contains market metadata, one-second DOGE RTI observations, order-book frames, and public trades. The implementation was checked against exchange-recorded targets. The alternative boundary-including window matched substantially less often. Primary backtests use exchange-recorded targets at both ends, while local reconstruction is used only for implementation validation and data-quality checks.

## Walk-forward probability testing

Forecasts are generated at 30, 60, 120, 300, and 600 seconds before expiration. For each test day:

- Only markets closed before that UTC day may be used for calibration.
- At least seven earlier days are required.
- At least 400 training rows are required.
- The test day's outcomes remain unseen until scoring.

Models are evaluated primarily using log loss because it evaluates probability quality and heavily punishes incorrect confident forecasts. Brier score, calibration error, AUC, and classification accuracy are retained as secondary diagnostics in the generated artifacts.

Kalshi market midpoints are included as a forecast benchmark when historical data is available. They are not treated as executable.

## From probability to quote

The probability forecast itself is not inventory-aware. The quoting layer converts model fair value $p_t$ into an inventory-adjusted reservation value

$$
r_t=\min\!\left(0.999,\max\!\left(0.001,p_t-\lambda q_t\right)\right),
$$

where $q_t$ is current YES inventory and $\lambda$ is the inventory-skew coefficient. A long YES position lowers the reservation value, reducing the maximum acceptable bid and the minimum acceptable ask. This discourages the strategy from buying more YES and makes it easier to sell existing inventory. A short YES position has the opposite effect.

The 1x tests use $\lambda=0.002$ with one-contract orders and a 10-contract inventory limit. The 10x tests use $\lambda=0.0002$ with ten-contract orders and a 100-contract inventory limit. At the inventory limit, the reservation-value adjustment is therefore two cents in either case. One full order changes the reservation value by 0.2 cent in either size setting.

The minimum model-to-quote distance is the minimum acceptable separation between the reservation value and a market price the strategy will join. For minimum distance $e$, the strategy joins bids only at or below $r_t-e$ and asks only at or above $r_t+e$. For example, using the one-cent setting with a 60-cent reservation value, the maximum bid is 59 cents and the minimum ask is 61 cents. The actual reservation-value-to-quote distance can be larger than the configured minimum.

## Event replay

Historical snapshots, deltas, and trades are processed in exchange-time order. Orders become active after 250 milliseconds of simulated placement latency. Desired cancellations are also delayed by 250 milliseconds, meaning a stale quote remains fillable during the cancellation interval. The last-minute replay quotes from $T-60$ through $T-15$; the full-market replay quotes from $T-900$ through $T-15$.

The replay reports two main fill assumptions.

### Trade-through

A hypothetical resting order fills only when an observed liquidity-taking trade reaches or passes the resting price. Merely appearing at the best bid or ask is insufficient. Fill quantity is capped by the liquidity-taking quantity actually observed.

This is a stricter robustness test, but it can miss legitimate at-price fills.

### Queue-aware

The displayed quantity already resting at the order price becomes queue ahead. Subsequent liquidity-taking volume first consumes the queue. Only remaining observed volume can fill the hypothetical order.

Queue-aware replay is more realistic than assuming every touch fills, but it is still approximate because hidden liquidity, private order acknowledgements, and exact queue priority are unavailable.

## P&L accounting

If the final YES outcome is $y\in \{0,1\}$, then buying $x$ contracts at price $a$ produces

$$\Pi_{\text{buy}} = x(y-a) - \text{fees}$$

and similarly selling $x$ contracts at price $b$ produces

$$\Pi_{\text{sell}} = x(b-y) - \text{fees}$$

Residual inventory is settled at the binary outcome. The replay also records how fair value changes one and five seconds after each fill to measure short-term adverse selection. Maximum inventory, drawdown, fill timing, and daily P&L are also recorded.

The tested historical period used zero base maker fees. Future fee schedules may differ.

## Statistical treatment

Markets within the same day share volatility, liquidity, and model state. They are not treated as thousands of independent trials. Confidence intervals resample complete UTC-day P&L blocks.

This addresses within-day dependence but not every source of research selection. Results from several edges, sizes, and fill models are related comparisons, so unadjusted intervals should not be interpreted as proof after choosing the best-performing variant.

## Limitations

The replay cannot establish the following points:

- A hypothetical order may have changed the behavior of the market.
- Displayed depth does not guarantee live queue priority.
- Private order acknowledgements are unavailable.
- Historical liquidity does not guarantee future capacity.
- Infrastructure limits are only partially represented.
