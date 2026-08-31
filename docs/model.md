# Probability Model

## Settlement Definition

KXDOGE15M is a binary market based on the arithmetic average of the CF Benchmarks DOGE RTI during the 60 seconds preceding expiration. If $T$ is the expiration and $K$ is Kalshi's published strike then the expiration value is

$$A_T = \frac{1}{60}\sum_{j=1}^{60} S_{T-j}$$

where $S_t$ is the current DOGE RTI at time $t$, so the settlement grid is $[T-60,T)$. YES then pays one dollar when $A_T \geq K$ since the market only requires the average to be at least the strike. The goal of the model is to estimate
$$p_t = \mathbb P (A_T \geq K | \mathcal F_t)$$
where $\mathcal F_t$ is all the available information at time $t$. Ignoring fees, this probability is also the expected dollar settlement value of one YES contract.

## Separate known and unknown settlement values

An ordinary digital option depends only on one terminal price, but KXDOGE15M depends on 60 prices. During the final minute some of those prices are already fixed. Writing the settlement sum as
$$60 A_T = C_t + R_t$$
where $C_t$ is the sum of RTI values in the final 60 second window observed at timestamps less than or equal to $t$ and $R_t$ is the sum of RTI values not yet observed at time $t$. The payoff condition stated in the previous section then becomes
$$R_t \geq B_t, \quad  B_t = 60K - C_t$$
Therefore, the model now aims to determine
$$p_t = \mathbb P (R_t \geq B_t | \mathcal F_t)$$

## Modelling future RTI path

The structural model uses a local geometric Brownian motion approximation. At each observation, future RTI observations are modelled as geometric Brownian motion with zero drift in the price level, whose volatility is estimated from the previous 300 seconds. This assumption is only used over short horizons. Empirical residual calibration compensates for recurring deviations from the GBM approximation. $S_t$ satisfies, for Brownian Motion $W$

$$dS_t = \sigma S_t \, dW_t$$

At time $t$, the future RTI at time $t+\tau$ is therefore

$$S_{t+\tau} = S_t\exp\left(\hat\sigma_t \Delta W_\tau - \frac{\hat\sigma_t^2}{2}\tau\right)$$

where $\Delta W_\tau = W_{t+\tau}-W_t$ is the future Brownian increment.

The volatility is estimated from 300-second log returns:

$$\hat\sigma_t^2 = \frac{\sum_k (\Delta \log S_k)^2}{\sum_k \Delta t_k}$$

## Correlation between Samples

The remaining one-second observations are not independent. They share the same price path. At time $t$, at future times $t+ \tau_i$ and $t + \tau_j$, the covariance is then found to be

$$\text{Cov}_t(S_{t+\tau_i},S_{t+\tau_j}) = S_t^2 (e^{\sigma_t^2 \min(\tau_i,\tau_j)} - 1)$$

If there are $n_t$ observations remaining, the expected value of the sum is then

$$M_t =\mathbb{E}_t[R_t] = n_t S_t$$

and the variance is

$$V_t = \operatorname{Var}_t(R_t) = \sum_{i=1}^{n_t} \sum_{j=1}^{n_t} \text{Cov}_t (S_{t+\tau_i}, S_{t+\tau_j})$$

## Approximating $R_t$

If there are $n_t$ observations remaining till contract expiry, $R_t$ is the sum of correlated lognormal variables. This then has form:

$$R_t = S_t\sum_{i=1}^{n_t}\exp\left(\sigma_t W_{\tau_i}-\frac{\sigma_t^2}{2}\tau_i \right)$$

The correlated lognormal sum has no convenient closed-form CDF, so moment matching is used to approximate it as lognormal. Let $\hat R_t = e^{Y_t}$ where $Y_t \sim \mathcal N(m_t,s_t^2)$. Then suppose

$$R_t \approx \hat R_t $$

Then

$$\mathbb{E}_t[\hat R_t] = \exp\left(m_t + \frac{s_t^2}{2}\right)$$

and
$$\operatorname{Var}_t(\hat R_t) = (e^{s_t^2}-1)e^{2m_t + s_t^2} = (e^{s_t^2}-1)(\mathbb{E}_t[\hat R_t])^2$$

Then imposing $\mathbb{E}_t[R_t] = \mathbb{E}_t[\hat R_t]$ and $\operatorname{Var}_t(\hat R_t) = V_t$

$$\begin{align*}\operatorname{Var}_t(\hat R_t) = V_t & \implies V_t = M_t^2 (e^{s_t^2}-1)
\\ & \implies s_t^2 = \log \left(1+ \frac{V_t}{M_t^2}\right)\end{align*}$$

and

$$\begin{align*}\mathbb{E}_t[\hat R_t] = M_t & \implies m_t = \log (M_t) - \frac{s_t^2}{2}\end{align*}$$

If $B_t \leq 0$, YES is already certain. If no samples remain and $B_t>0$, YES is impossible. Otherwise the probability is given as follows:

$$\begin{align*}p_t & = \mathbb P_t(R_t \geq B_t)
\\ & \approx \mathbb{P}_t(Y_t \geq \log B_t)
\\ & = \mathbb{P}_t\left(\frac{Y_t - m_t}{s_t} \geq \frac{\log B_t - m_t}{s_t}\right)
\\ & = 1 - \Phi\left(\frac{\log B_t - m_t}{s_t}
\right)
\\ & = \Phi\left(\frac{m_t-\log B_t}{s_t}\right)\end{align*}$$

Where $\Phi$ is the standard normal CDF. The implementation clips probabilities to $[0.001,0.999]$.

## Empirical calibration using residuals

For a completed market, we define the standardized residual as

$$\varepsilon_t = \frac{\log R_t^{\text{realized}} - m_t}{s_t}$$

Let

$$z_t = \frac{m_t - \log B_t}{s_t}$$

Then YES occurs when $\varepsilon_t \geq -z_t$. Therefore the empirically calibrated probability is

$$p_t^{\text{cal}} = 1-\hat F_h(-z_t)$$

Where $\hat F_h$ is the historical residual CDF for time horizon $h$, implemented with finite-sample smoothing. Residuals are maintained separately at 30, 60, 120, 300, and 600 seconds before expiration. Each test day uses only residuals from markets that closed before that UTC day.
