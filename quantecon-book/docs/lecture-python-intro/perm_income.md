---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python
    language: python3
    name: python3
---


<a id='perm-income'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>

<!-- #region -->
# The Permanent Income Model


<a id='index-1'></a>
<!-- #endregion -->

## Contents

- [The Permanent Income Model](#The-Permanent-Income-Model)  
  - [Overview](#Overview)  
  - [The Savings Problem](#The-Savings-Problem)  
  - [Alternative Representations](#Alternative-Representations)  
  - [Two Classic Examples](#Two-Classic-Examples)  
  - [Further Reading](#Further-Reading)  
  - [Appendix: The Euler Equation](#Appendix:-The-Euler-Equation)  


In addition to what’s in Anaconda, this lecture will need the following libraries:

```python3 hide-output=true
!pip install --upgrade quantecon
```

<!-- #region -->
## Overview

This lecture describes a rational expectations version of the famous permanent income model of Milton Friedman [[Fri56]](https://python-programming.quantecon.org/zreferences.html#friedman1956).

Robert Hall cast Friedman’s model within a linear-quadratic setting [[Hal78]](https://python-programming.quantecon.org/zreferences.html#hall1978).

Like Hall, we formulate an infinite-horizon linear-quadratic savings problem.

We use the model as a vehicle for illustrating

- alternative formulations of the *state* of a dynamic system  
- the idea of *cointegration*  
- impulse response functions  
- the idea that changes in consumption are useful as predictors of movements in income  


Background readings on the linear-quadratic-Gaussian permanent income model are Hall’s  [[Hal78]](https://python-programming.quantecon.org/zreferences.html#hall1978)  and chapter 2 of  [[LS18]](https://python-programming.quantecon.org/zreferences.html#ljungqvist2012).

Let’s start with some imports
<!-- #endregion -->

```python3 hide-output=false
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import random
from numba import njit
```

<!-- #region -->
## The Savings Problem


<a id='index-2'></a>
In this section, we state and solve the savings and consumption problem faced
by the consumer.
<!-- #endregion -->

<!-- #region -->
### Preliminaries

We use a class of stochastic processes called
[martingales](https://en.wikipedia.org/wiki/Martingale_%28probability_theory%29).

A discrete-time martingale is a stochastic process (i.e., a  sequence of random variables)
$ \{X_t\} $ with finite mean at each $ t $ and satisfying

$$
\mathbb{E}_t [X_{t+1} ] = X_t, \qquad t = 0, 1, 2, \ldots
$$

Here $ \mathbb{E}_t := \mathbb{E}[ \cdot \,|\, \mathcal{F}_t] $ is a conditional mathematical  expectation conditional on the time $ t $
*information set* $ \mathcal{F}_t $.

The latter is just a collection of random variables that the modeler declares
to be visible at $ t $.

- When not explicitly defined, it is usually understood that $ \mathcal{F}_t = \{X_t, X_{t-1}, \ldots, X_0\} $.  


Martingales have the feature that the history of past outcomes provides no predictive power for changes between current and future outcomes.

For example, the current wealth of a gambler engaged in a “fair game” has this
property.

One common class of martingales is the family of *random walks*.

A **random walk** is  a stochastic process $ \{X_t\} $ that satisfies

$$
X_{t+1} = X_t + w_{t+1}
$$

for some IID zero mean *innovation* sequence $ \{w_t\} $.

Evidently, $ X_t $ can also be expressed as

$$
X_t = \sum_{j=1}^t w_j + X_0
$$

Not every martingale arises as a random walk (see, for example, [Wald’s martingale](https://en.wikipedia.org/wiki/Wald%27s_martingale)).
<!-- #endregion -->

<!-- #region -->
### The Decision Problem

A consumer has preferences over consumption streams that are ordered by the utility functional


<a id='equation-sprob1'></a>
$$
\mathbb{E}_0 \left[ \sum_{t=0}^\infty \beta^t u(c_t) \right] \tag{1}
$$

where

- $ \mathbb{E}_t $ is the mathematical expectation conditioned on the consumer’s time $ t $ information  
- $ c_t $ is time $ t $ consumption  
- $ u $ is a strictly concave one-period utility function  
- $ \beta \in (0,1) $ is a discount factor  


The consumer maximizes [(1)](#equation-sprob1) by choosing a consumption, borrowing plan $ \{c_t, b_{t+1}\}_{t=0}^\infty $ subject to the sequence of budget constraints


<a id='equation-sprob2'></a>
$$
c_t + b_t = \frac{1}{1 + r} b_{t+1} +  y_t   \quad t \geq 0 \tag{2}
$$

Here

- $ y_t $ is an exogenous endowment process.  
- $ r > 0 $ is a time-invariant risk-free net interest rate.  
- $ b_t $ is one-period risk-free debt maturing at $ t $.  


The consumer also faces initial conditions $ b_0 $ and $ y_0 $, which can be fixed or random.
<!-- #endregion -->

<!-- #region -->
### Assumptions

For the remainder of this lecture, we follow Friedman and Hall in assuming that $ (1 + r)^{-1} = \beta $.

Regarding the endowment process, we assume it has the [state-space representation](https://python-programming.quantecon.org/linear_models.html)


<a id='equation-sprob15ab'></a>
$$
\begin{aligned}
    z_{t+1} & = A z_t + C w_{t+1}
    \\
    y_t & = U  z_t
\end{aligned} \tag{3}
$$

where

- $ \{w_t\} $ is an IID vector process with $ \mathbb{E} w_t = 0 $ and $ \mathbb{E} w_t w_t' = I $.  
- The [spectral radius](https://python-programming.quantecon.org/linear_algebra.html#la-neumann-remarks) of $ A $ satisfies $ \rho(A) < \sqrt{1/\beta} $.  
- $ U $ is a selection vector that pins down $ y_t $ as a particular linear combination of components of $ z_t $.  


The restriction on $ \rho(A) $ prevents income from growing so fast that discounted geometric sums of some quadratic forms to be described below become infinite.

Regarding preferences, we assume the quadratic utility function

$$
u(c_t) =  - (c_t - \gamma)^2
$$

where $ \gamma $ is a bliss level of consumption

>**Note**
>
>Along with this quadratic utility specification, we allow consumption to be negative.  However, by choosing parameters appropriately, we can make the probability that the model generates negative consumption paths over finite time horizons as low as desired.

Finally, we impose the *no Ponzi scheme* condition


<a id='equation-sprob3'></a>
$$
\mathbb{E}_0 \left[ \sum_{t=0}^\infty \beta^t b_t^2 \right] < \infty \tag{4}
$$

This condition rules out an always-borrow scheme that would allow the consumer to enjoy bliss consumption forever.
<!-- #endregion -->

<!-- #region -->
### First-Order Conditions

First-order conditions for maximizing [(1)](#equation-sprob1) subject to [(2)](#equation-sprob2) are


<a id='equation-sprob4'></a>
$$
\mathbb{E}_t [u'(c_{t+1})] = u'(c_t) , \qquad t = 0, 1, \ldots \tag{5}
$$

These optimality conditions are also known as  *Euler equations*.

If you’re not sure where they come from, you can find a proof sketch in the
[appendix](#perm-income-appendix).

With our quadratic preference specification, [(5)](#equation-sprob4) has the striking implication that consumption follows a martingale:


<a id='equation-sprob5'></a>
$$
\mathbb{E}_t [c_{t+1}] = c_t \tag{6}
$$

(In fact, quadratic preferences are *necessary* for this conclusion <sup><a href=#f2 id=f2-link>[1]</a></sup>)

One way to interpret [(6)](#equation-sprob5) is that consumption will change only when
“new information” about permanent income is revealed.

These ideas will be clarified below.


<a id='odr-pi'></a>
<!-- #endregion -->

<!-- #region -->
### The Optimal Decision Rule

Now let’s deduce the optimal decision rule <sup><a href=#fod id=fod-link>[2]</a></sup>

>**Note**
>
>One way to solve the consumer’s problem is to apply *dynamic programming*
as in [this lecture](https://python-programming.quantecon.org/lqcontrol.html).  We do this later. But first we use
an alternative approach that is revealing and shows the work that dynamic
programming does for us behind the scenes.

In doing so, we need to combine

1. the optimality condition [(6)](#equation-sprob5)  
1. the period-by-period budget constraint [(2)](#equation-sprob2), and  
1. the boundary condition [(4)](#equation-sprob3)  


To accomplish this, observe first that [(4)](#equation-sprob3) implies $ \lim_{t \to \infty} \beta^{\frac{t}{2}} b_{t+1}= 0 $.

Using this restriction on the debt path and solving [(2)](#equation-sprob2) forward yields


<a id='equation-sprob6'></a>
$$
b_t = \sum_{j=0}^\infty \beta^j (y_{t+j} - c_{t+j}) \tag{7}
$$

Take conditional expectations on both sides of [(7)](#equation-sprob6) and use the martingale property of consumption and the *law of iterated expectations* to deduce


<a id='equation-sprob7'></a>
$$
b_t = \sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}] - \frac{c_t}{1-\beta} \tag{8}
$$

Expressed in terms of $ c_t $ we get


<a id='equation-sprob8'></a>
$$
c_t
 = (1-\beta) \left[ \sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}] - b_t\right]
 = {r \over 1+r} \left[ \sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}] - b_t\right] \tag{9}
$$

where the last equality uses $ (1 + r) \beta = 1 $.

These last two equations assert that consumption equals *economic income*

- **financial wealth** equals $ -b_t $  
- **non-financial wealth** equals $ \sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}] $  
- **total wealth** equals the sum of financial and non-financial wealth  
- a **marginal propensity to consume out of total wealth** equals the  interest factor $ \frac{r}{1+r} $  
- **economic income** equals  
  - a constant marginal propensity to consume  times the sum of non-financial wealth and financial wealth  
  - the amount the consumer can consume while leaving its wealth intact  
<!-- #endregion -->

<!-- #region -->
#### Responding to the State

The *state* vector confronting the consumer at $ t $ is $ \begin{bmatrix} b_t & z_t \end{bmatrix} $.

Here

- $ z_t $ is an *exogenous* component, unaffected by consumer behavior.  
- $ b_t $ is an *endogenous* component (since it depends on the decision rule).  


Note that $ z_t $ contains all variables useful for forecasting the consumer’s future endowment.

It is plausible that current decisions $ c_t $ and $ b_{t+1} $ should
be expressible as functions of $ z_t $ and $ b_t $.

This is indeed the case.

In fact, from [this discussion](https://python-programming.quantecon.org/linear_models.html#lm-fgs), we see that

$$
\sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}]
= \mathbb{E}_t \left[ \sum_{j=0}^\infty \beta^j y_{t+j} \right]
= U(I - \beta A)^{-1} z_t
$$

Combining this with [(9)](#equation-sprob8) gives


<a id='equation-pi-cpa'></a>
$$
c_t
 = {r \over 1+r}
     \left[
         U(I - \beta A)^{-1} z_t - b_t
     \right] \tag{10}
$$

Using this equality to eliminate $ c_t $ in the budget constraint [(2)](#equation-sprob2) gives

$$
\begin{aligned}
    b_{t+1}
    & = (1 + r) (b_t + c_t - y_t)
    \\
    & = (1 + r) b_t + r [ U(I - \beta A)^{-1} z_t - b_t]  - (1+r) U z_t
    \\
    & = b_t +  U [ r(I - \beta A)^{-1}  - (1+r) I ]  z_t
    \\
    & = b_t +  U (I - \beta A)^{-1} (A - I)   z_t
\end{aligned}
$$

To get from the second last to the last expression in this chain of equalities is not trivial.

A key is to use the fact that $ (1 + r) \beta = 1 $ and $ (I - \beta A)^{-1} = \sum_{j=0}^{\infty} \beta^j A^j $.

We’ve now successfully written $ c_t $ and $ b_{t+1} $ as functions of $ b_t $ and $ z_t $.
<!-- #endregion -->

<!-- #region -->
#### A State-Space Representation

We can summarize our dynamics in the form of a linear state-space system governing consumption, debt and income:


<a id='equation-pi-ssr'></a>
$$
\begin{aligned}
  z_{t+1} & = A z_t + C w_{t+1} \\
  b_{t+1} & = b_t + U [ (I -\beta A)^{-1} (A - I) ] z_t \\
      y_t & = U z_t \\
      c_t & = (1-\beta) [ U(I-\beta A)^{-1} z_t - b_t ]
\end{aligned} \tag{11}
$$

To write this more succinctly, let

$$
x_t =
\begin{bmatrix}
    z_t\\
    b_t
\end{bmatrix},
\quad
\tilde A =
\begin{bmatrix}
    A & 0 \\
    U(I-\beta A)^{-1}(A-I) & 1
 \end{bmatrix},
 \quad
\tilde C =
\begin{bmatrix}
    C\\
    0
\end{bmatrix}
$$

and

$$
\tilde U =
\begin{bmatrix}
    U & 0 \\
    (1-\beta) U (I - \beta A)^{-1} & -(1-\beta)
\end{bmatrix}, \quad
\tilde y_t =
\begin{bmatrix}
      y_t\\
      c_t
  \end{bmatrix}
$$

Then we can express equation  [(11)](#equation-pi-ssr) as


<a id='equation-pi-stsp'></a>
$$
\begin{aligned}
  x_{t+1} & = \tilde A x_t + \tilde C w_{t+1} \\
  \tilde y_t & = \tilde U x_t
\end{aligned} \tag{12}
$$

We can use the following formulas from [linear state space models](https://python-programming.quantecon.org/linear_models.html) to compute population mean $ \mu_t = \mathbb{E}    x_t $ and covariance $ \Sigma_t := \mathbb{E} [ (x_t - \mu_t) (x_t - \mu_t)'] $


<a id='equation-lss-mut-perm-income'></a>
$$
\mu_{t+1} = \tilde A \mu_t
\quad \text{with} \quad \mu_0 \text{ given} \tag{13}
$$


<a id='equation-eqsigmalaw'></a>
$$
\Sigma_{t+1}  = \tilde A  \Sigma_t \tilde A' + \tilde C \tilde C'
\quad \text{with} \quad \Sigma_0 \text{ given} \tag{14}
$$

We can then compute the mean and covariance of $ \tilde y_t $ from


<a id='equation-eqymoments'></a>
$$
\begin{aligned}
\mu_{y,t} = \tilde U \mu_t \\
\Sigma_{y,t} = \tilde U \Sigma_t \tilde U'
\end{aligned} \tag{15}
$$
<!-- #endregion -->

#### A Simple Example with IID Income

To gain some preliminary intuition on the implications of [(11)](#equation-pi-ssr), let’s look at a highly stylized example where income is just IID.

(Later examples will investigate more realistic income streams)

In particular, let $ \{w_t\}_{t = 1}^{\infty} $ be IID and scalar standard normal, and let

$$
z_t =
\begin{bmatrix}
    z^1_t \\
    1
\end{bmatrix},
\quad
A =
\begin{bmatrix}
    0 & 0 \\
    0 & 1
\end{bmatrix},
\quad
U =
\begin{bmatrix}
    1 & \mu
\end{bmatrix},
\quad
C =
\begin{bmatrix}
    \sigma \\
    0
\end{bmatrix}
$$

Finally, let $ b_0 = z^1_0 = 0 $.

Under these assumptions, we have $ y_t = \mu + \sigma w_t \sim N(\mu, \sigma^2) $.

Further, if you work through the state space representation, you will see that

$$
\begin{aligned}
    b_t & = - \sigma \sum_{j=1}^{t-1} w_j
    \\
    c_t & = \mu + (1 - \beta) \sigma \sum_{j=1}^t w_j
\end{aligned}
$$

Thus income is IID and debt and consumption are both Gaussian random walks.

Defining assets as $ -b_t $, we see that assets are just the cumulative sum of unanticipated incomes prior to the present date.

The next figure shows a typical realization with $ r = 0.05 $, $ \mu = 1 $, and $ \sigma = 0.15 $

```python3 hide-output=false
r = 0.05
β = 1 / (1 + r)
σ = 0.15
μ = 1
T = 60

@njit
def time_path(T):
    w = np.random.randn(T+1)  # w_0, w_1, ..., w_T
    w[0] = 0
    b = np.zeros(T+1)
    for t in range(1, T+1):
        b[t] = w[1:t].sum()
    b = -σ * b
    c = μ + (1 - β) * (σ * w - b)
    return w, b, c

w, b, c = time_path(T)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(μ + σ * w, 'g-', label="Non-financial income")
ax.plot(c, 'k-', label="Consumption")
ax.plot( b, 'b-', label="Debt")
ax.legend(ncol=3, mode='expand', bbox_to_anchor=(0., 1.02, 1., .102))
ax.grid()
ax.set_xlabel('Time')

plt.show()
```

Observe that consumption is considerably smoother than income.

The figure below shows the consumption paths of 250 consumers with independent income streams

```python3 hide-output=false
fig, ax = plt.subplots(figsize=(10, 6))

b_sum = np.zeros(T+1)
for i in range(250):
    w, b, c = time_path(T)  # Generate new time path
    rcolor = random.choice(('c', 'g', 'b', 'k'))
    ax.plot(c, color=rcolor, lw=0.8, alpha=0.7)

ax.grid()
ax.set(xlabel='Time', ylabel='Consumption')

plt.show()
```

## Alternative Representations

In this section, we shed more light on the evolution of savings, debt and
consumption by representing their dynamics in several different ways.

<!-- #region -->
### Hall’s Representation


<a id='index-3'></a>
Hall [[Hal78]](https://python-programming.quantecon.org/zreferences.html#hall1978) suggested an insightful way to summarize the implications of LQ permanent income theory.

First, to represent the solution for $ b_t $, shift [(9)](#equation-sprob8) forward one period and eliminate $ b_{t+1} $ by using [(2)](#equation-sprob2) to obtain

$$
c_{t+1} = (1-\beta)\sum_{j=0}^\infty \beta^j  \mathbb{E}_{t+1} [y_{t+j+1}] -
(1-\beta) \left[ \beta^{-1} (c_t + b_t - y_t) \right]
$$

If we add and subtract $ \beta^{-1} (1-\beta) \sum_{j=0}^\infty \beta^j \mathbb{E}_t y_{t+j} $ from the right side of the preceding equation and rearrange, we obtain


<a id='equation-sprob11'></a>
$$
c_{t+1} - c_t = (1-\beta) \sum_{j=0}^\infty \beta^j
    \left\{ \mathbb{E}_{t+1} [y_{t+j+1}] - \mathbb{E}_t [y_{t+j+1}] \right\} \tag{16}
$$

The right side is the time $ t+1 $ *innovation to the expected present value* of the endowment process $ \{y_t\} $.

We can represent the optimal decision rule for $ (c_t, b_{t+1}) $ in the form of [(16)](#equation-sprob11) and [(8)](#equation-sprob7), which we repeat:


<a id='equation-sprob7aa'></a>
$$
b_t = \sum_{j=0}^\infty \beta^j \mathbb{E}_t [y_{t+j}] - {1 \over 1-\beta} c_t \tag{17}
$$

Equation [(17)](#equation-sprob7aa) asserts that the consumer’s debt due at $ t $ equals the expected present value of its endowment minus the expected present value of its consumption stream.

A high debt thus indicates a large expected present value of surpluses $ y_t - c_t $.

Recalling again our discussion on [forecasting geometric sums](https://python-programming.quantecon.org/linear_models.html#lm-fgs), we have

$$
\begin{aligned}
    \mathbb{E}_t \sum_{j=0}^\infty \beta^j y_{t+j} &= U (I-\beta A)^{-1} z_t \\
    \mathbb{E}_{t+1} \sum_{j=0}^\infty \beta^j y_{t+j+1} & = U (I -\beta A)^{-1} z_{t+1} \\
    \mathbb{E}_t \sum_{j=0}^\infty \beta^j y_{t+j+1} & = U (I - \beta A)^{-1} A z_t
\end{aligned}
$$

Using  these formulas together with [(3)](#equation-sprob15ab) and substituting  into [(16)](#equation-sprob11) and [(17)](#equation-sprob7aa)  gives the following representation for the consumer’s optimum decision rule:


<a id='equation-sprob16abcd'></a>
$$
\begin{aligned}
    c_{t+1} & = c_t + (1-\beta) U  (I-\beta A)^{-1} C w_{t+1} \\
    b_t & = U (I-\beta A)^{-1} z_t - {1 \over 1-\beta} c_t \\
    y_t & = U z_t \\
    z_{t+1} & = A z_t + C w_{t+1}
\end{aligned} \tag{18}
$$

Representation [(18)](#equation-sprob16abcd) makes clear that

- The state can be taken as $ (c_t, z_t) $.  
  
  - The endogenous part is $ c_t $ and the exogenous part is $ z_t $.  
  - Debt $ b_t $ has disappeared as a component of the state because it is encoded in $ c_t $.  
  
- Consumption is a random walk with innovation $ (1-\beta) U  (I-\beta A)^{-1} C w_{t+1} $.  
  
  - This is a more explicit representation of the martingale result in [(6)](#equation-sprob5).  
  



<a id='coint-pi'></a>
<!-- #endregion -->

<!-- #region -->
### Cointegration

Representation [(18)](#equation-sprob16abcd) reveals that the joint process $ \{c_t, b_t\} $ possesses the property that Engle and Granger [[EG87]](https://python-programming.quantecon.org/zreferences.html#englegranger1987) called [cointegration](https://en.wikipedia.org/wiki/Cointegration).

Cointegration is a tool that allows us to apply powerful results from the theory of stationary stochastic processes to (certain transformations of) nonstationary models.

To apply cointegration in the present context, suppose that $ z_t $ is asymptotically stationary <sup><a href=#fn-as id=fn-as-link>[4]</a></sup>.

Despite this, both $ c_t $ and $ b_t $ will be non-stationary because they have unit roots (see [(11)](#equation-pi-ssr) for $ b_t $).

Nevertheless, there is a linear combination of $ c_t, b_t $ that *is* asymptotically stationary.

In particular, from the second equality in [(18)](#equation-sprob16abcd) we have


<a id='equation-pi-spr'></a>
$$
(1-\beta) b_t + c_t = (1 - \beta) U (I-\beta A)^{-1} z_t \tag{19}
$$

Hence the linear combination $ (1-\beta) b_t + c_t $ is asymptotically stationary.

Accordingly, Granger and Engle would call $ \begin{bmatrix} (1-\beta) & 1 \end{bmatrix} $ a **cointegrating vector** for the state.

When applied to the nonstationary vector process $ \begin{bmatrix} b_t  & c_t \end{bmatrix}' $, it yields a process that is asymptotically stationary.

Equation [(19)](#equation-pi-spr) can be rearranged to take the form


<a id='equation-sprob77'></a>
$$
(1-\beta) b_t + c_t = (1-\beta) \mathbb{E}_t \sum_{j=0}^\infty \beta^j y_{t+j} \tag{20}
$$

Equation [(20)](#equation-sprob77)  asserts that the *cointegrating residual*  on the left side equals the conditional expectation of the geometric sum of future incomes on the right <sup><a href=#f8 id=f8-link>[6]</a></sup>.
<!-- #endregion -->

<!-- #region -->
### Cross-Sectional Implications

Consider again [(18)](#equation-sprob16abcd), this time in light of our discussion of
distribution dynamics in the [lecture on linear systems](https://python-programming.quantecon.org/linear_models.html).

The dynamics of $ c_t $ are given by


<a id='equation-pi-crw'></a>
$$
c_{t+1} = c_t + (1-\beta) U  (I-\beta A)^{-1} C w_{t+1} \tag{21}
$$

or

$$
c_t = c_0 + \sum_{j=1}^t \hat w_j
\quad \text{for} \quad
\hat w_{t+1} := (1-\beta) U  (I-\beta A)^{-1} C w_{t+1}
$$

The unit root affecting $ c_t $ causes the time $ t $ variance of $ c_t $ to grow linearly with $ t $.

In particular, since $ \{ \hat w_t \} $ is IID, we have


<a id='equation-pi-vt'></a>
$$
\mathrm{Var}[c_t] = \mathrm{Var}[c_0] + t \, \hat \sigma^2 \tag{22}
$$

where

$$
\hat \sigma^2 := (1-\beta)^2 U  (I-\beta A)^{-1} CC' (I-\beta A')^{-1} U'
$$

When $ \hat \sigma > 0 $, $ \{c_t\} $ has no asymptotic distribution.

Let’s consider what this means for a cross-section of ex-ante identical consumers born at time $ 0 $.

Let the distribution of $ c_0 $ represent the cross-section of initial consumption values.

Equation [(22)](#equation-pi-vt) tells us that the variance of $ c_t $ increases over time at a rate proportional to $ t $.

A number of different studies have investigated this prediction and found some support for it
(see, e.g., [[DP94]](https://python-programming.quantecon.org/zreferences.html#deatonpaxton1994), [[STY04]](https://python-programming.quantecon.org/zreferences.html#sty2004)).
<!-- #endregion -->

### Impulse Response Functions

Impulse response functions measure responses  to various  impulses (i.e., temporary shocks).

The impulse response function of $ \{c_t\} $ to the innovation $ \{w_t\} $ is a box.

In particular, the response of $ c_{t+j} $ to a unit increase in the innovation $ w_{t+1} $ is $ (1-\beta) U (I -\beta A)^{-1} C $ for all $ j \geq 1 $.

<!-- #region -->
### Moving Average Representation

It’s useful to express the innovation to the expected present value of the endowment process in terms of a moving average representation for income $ y_t $.

The endowment process defined by [(3)](#equation-sprob15ab) has the moving average representation


<a id='equation-sprob12'></a>
$$
y_{t+1} = d(L) w_{t+1} \tag{23}
$$

where

- $ d(L) = \sum_{j=0}^\infty d_j L^j $ for some sequence $ d_j $, where $ L $ is the lag operator <sup><a href=#f4 id=f4-link>[3]</a></sup>  
- at time $ t $, the consumer has an information set <sup><a href=#f5 id=f5-link>[5]</a></sup> $ w^t = [w_t, w_{t-1}, \ldots ] $  


Notice that

$$
y_{t+j} - \mathbb{E}_t [y_{t+j}] = d_0 w_{t+j} + d_1 w_{t+j-1} + \cdots + d_{j-1} w_{t+1}
$$

It follows that


<a id='equation-sprob120'></a>
$$
\mathbb{E}_{t+1} [y_{t+j}] - \mathbb{E}_t [y_{t+j}] = d_{j-1} w_{t+1} \tag{24}
$$

Using [(24)](#equation-sprob120) in [(16)](#equation-sprob11) gives


<a id='equation-sprob13'></a>
$$
c_{t+1} - c_t = (1-\beta) d(\beta) w_{t+1} \tag{25}
$$

The object $ d(\beta) $ is the **present value of the moving average coefficients** in the representation for the endowment process $ y_t $.


<a id='sub-classic-consumption'></a>
<!-- #endregion -->

## Two Classic Examples

We illustrate some of the preceding ideas with two examples.

In both examples, the endowment follows the process $ y_t = z_{1t} + z_{2t} $ where

$$
\begin{bmatrix}
  z_{1 t+1} \\
  z_{2t+1}
\end{bmatrix} =
\begin{bmatrix}
  1 & 0 \\
  0 & 0
\end{bmatrix}
\begin{bmatrix}
  z_{1t} \\
  z_{2t}
\end{bmatrix} + \begin{bmatrix}
      \sigma_1 & 0 \\
      0 & \sigma_2
  \end{bmatrix}
  \begin{bmatrix}
      w_{1t+1} \\
      w_{2t+1}
  \end{bmatrix}
$$

Here

- $ w_{t+1} $ is an IID $ 2 \times 1 $ process distributed as $ N(0,I) $.  
- $ z_{1t} $ is a permanent component of $ y_t $.  
- $ z_{2t} $ is a purely transitory component of $ y_t $.  

<!-- #region -->
### Example 1

Assume as before that the consumer observes the state $ z_t $ at time $ t $.

In view of [(18)](#equation-sprob16abcd) we have


<a id='equation-consexample1'></a>
$$
c_{t+1} - c_t = \sigma_1 w_{1t+1} + (1-\beta) \sigma_2 w_{2t+1} \tag{26}
$$

Formula [(26)](#equation-consexample1) shows how an increment $ \sigma_1 w_{1t+1} $ to the permanent component of income $ z_{1t+1} $ leads to

- a permanent one-for-one increase in consumption and  
- no increase in savings $ -b_{t+1} $  


But the purely transitory component of income $ \sigma_2 w_{2t+1} $ leads to a permanent increment in consumption by a fraction $ 1-\beta $ of transitory income.

The remaining fraction $ \beta $ is saved, leading to a permanent increment in $ -b_{t+1} $.

Application of the formula for debt in [(11)](#equation-pi-ssr) to this example shows that


<a id='equation-consexample1a'></a>
$$
b_{t+1} - b_t = - z_{2t} = - \sigma_2 w_{2t} \tag{27}
$$

This confirms that none of $ \sigma_1 w_{1t} $ is saved, while all of $ \sigma_2 w_{2t} $ is saved.

The next figure illustrates these very different reactions to transitory and
permanent income shocks using impulse-response functions
<!-- #endregion -->

```python3 hide-output=false
r = 0.05
β = 1 / (1 + r)
S = 5   # Impulse date
σ1 = σ2 = 0.15

@njit
def time_path(T, permanent=False):
    "Time path of consumption and debt given shock sequence"
    w1 = np.zeros(T+1)
    w2 = np.zeros(T+1)
    b = np.zeros(T+1)
    c = np.zeros(T+1)
    if permanent:
        w1[S+1] = 1.0
    else:
        w2[S+1] = 1.0
    for t in range(1, T):
        b[t+1] = b[t] - σ2 * w2[t]
        c[t+1] = c[t] + σ1 * w1[t+1] + (1 - β) * σ2 * w2[t+1]
    return b, c


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
titles = ['transitory', 'permanent']

L = 0.175

for ax, truefalse, title in zip(axes, (True, False), titles):
    b, c = time_path(T=20, permanent=truefalse)
    ax.set_title(f'Impulse reponse: {title} income shock')
    ax.plot(c, 'g-', label="consumption")
    ax.plot(b, 'b-', label="debt")
    ax.plot((S, S), (-L, L), 'k-', lw=0.5)
    ax.grid(alpha=0.5)
    ax.set(xlabel=r'Time', ylim=(-L, L))

axes[0].legend(loc='lower right')

plt.tight_layout()
plt.show()
```

<!-- #region -->
### Example 2

Assume now that at time $ t $ the consumer observes $ y_t $, and its history up to $ t $, but not $ z_t $.

Under this assumption, it is appropriate to use an *innovation representation* to form $ A, C, U $ in [(18)](#equation-sprob16abcd).

The discussion in sections 2.9.1 and 2.11.3 of [[LS18]](https://python-programming.quantecon.org/zreferences.html#ljungqvist2012) shows that the pertinent state space representation for $ y_t $ is

$$
\begin{aligned}
  \begin{bmatrix}
    y_{t+1} \\
    a_{t+1}
  \end{bmatrix}
    & =
    \begin{bmatrix}
        1 & -(1 - K) \\
        0 & 0
    \end{bmatrix}
    \begin{bmatrix}
        y_t \\
        a_t
    \end{bmatrix} +
    \begin{bmatrix}
        1 \\
        1
    \end{bmatrix}
    a_{t+1}
    \\
    y_t
    & =
    \begin{bmatrix}
        1 & 0
    \end{bmatrix}
    \begin{bmatrix}
        y_t \\
        a_t
    \end{bmatrix}
\end{aligned}
$$

where

- $ K := $ the stationary Kalman gain  
- $ a_t := y_t - E [ y_t \,|\, y_{t-1}, \ldots, y_0] $  


In the same discussion in [[LS18]](https://python-programming.quantecon.org/zreferences.html#ljungqvist2012) it is shown that $ K \in [0,1] $ and that $ K $ increases as $ \sigma_1/\sigma_2 $ does.

In other words, $ K $ increases as the ratio of the standard deviation of the permanent shock to that of the transitory shock increases.

Please see  [first look at the Kalman filter](https://python-programming.quantecon.org/kalman.html).

Applying formulas [(18)](#equation-sprob16abcd) implies


<a id='equation-consexample2'></a>
$$
c_{t+1} - c_t = [1-\beta(1-K) ] a_{t+1} \tag{28}
$$

where the endowment process can now be represented in terms of the univariate innovation to $ y_t $ as


<a id='equation-incomemaar'></a>
$$
y_{t+1} - y_t = a_{t+1} - (1-K) a_t \tag{29}
$$

Equation [(29)](#equation-incomemaar) indicates that the consumer regards

- fraction $ K $ of an innovation $ a_{t+1} $ to $ y_{t+1} $ as *permanent*  
- fraction $ 1-K $ as purely transitory  


The consumer permanently increases his consumption by the full amount of his estimate of the permanent part of $ a_{t+1} $, but by only $ (1-\beta) $ times his estimate of the purely transitory part of $ a_{t+1} $.

Therefore, in total, he permanently increments his consumption by a fraction $ K + (1-\beta) (1-K) = 1 - \beta (1-K) $ of $ a_{t+1} $.

He saves the remaining fraction $ \beta (1-K) $.

According to equation [(29)](#equation-incomemaar), the first difference of income is a first-order moving average.

Equation  [(28)](#equation-consexample2) asserts that the first difference of consumption is IID.

Application of formula to this example shows that


<a id='equation-consexample1b'></a>
$$
b_{t+1} - b_t = (K-1) a_t \tag{30}
$$

This indicates how the fraction $ K $ of the innovation to $ y_t $ that is regarded as permanent influences the fraction of the innovation that is saved.
<!-- #endregion -->

<!-- #region -->
## Further Reading

The model described above significantly changed how economists think about
consumption.

While Hall’s model does a remarkably good job as a first approximation to consumption data, it’s widely believed that it doesn’t capture important aspects of some consumption/savings data.

For example, liquidity constraints and precautionary savings appear to be present sometimes.

Further discussion can be found in, e.g., [[HM82]](https://python-programming.quantecon.org/zreferences.html#hallmishkin1982), [[Par99]](https://python-programming.quantecon.org/zreferences.html#parker1999), [[Dea91]](https://python-programming.quantecon.org/zreferences.html#deaton1991), [[Car01]](https://python-programming.quantecon.org/zreferences.html#carroll2001).


<a id='perm-income-appendix'></a>
<!-- #endregion -->

## Appendix: The Euler Equation

Where does the first-order condition [(5)](#equation-sprob4) come from?

Here we’ll give a proof for the two-period case, which is representative of
the general argument.

The finite horizon equivalent of the no-Ponzi condition is that the agent
cannot end her life in debt, so $ b_2 = 0 $.

From the budget constraint [(2)](#equation-sprob2) we then have

$$
c_0 = \frac{b_1}{1 + r} - b_0 + y_0
\quad \text{and} \quad
c_1 = y_1 - b_1
$$

Here $ b_0 $ and $ y_0 $ are given constants.

Substituting these constraints into our two-period objective $ u(c_0) + \beta \mathbb{E}_0 [u(c_1)] $ gives

$$
\max_{b_1}
 \left\{
     u \left(\frac{b_1}{R} - b_0 + y_0 \right) +
     \beta \, \mathbb{E}_0 [ u (y_1 - b_1) ]
\right\}
$$

You will be able to verify that the first-order condition is

$$
u'(c_0) = \beta R  \,\mathbb{E}_0 [u'(c_1)]
$$

Using $ \beta R = 1 $ gives [(5)](#equation-sprob4) in the two-period case.

The proof for the general case is similar


**Footnotes**

<p><a id=f2 href=#f2-link><strong>[1]</strong></a> A linear marginal utility is essential for deriving [(6)](#equation-sprob5) from [(5)](#equation-sprob4).  Suppose instead that we had imposed the following more standard assumptions on the utility function: $ u'(c) >0, u''(c)<0, u'''(c) > 0 $ and required that $ c \geq 0 $.  The Euler equation remains [(5)](#equation-sprob4). But the fact that $ u''' <0 $ implies via Jensen’s inequality that $ \mathbb{E}_t [u'(c_{t+1})] >  u'(\mathbb{E}_t [c_{t+1}]) $.  This inequality together with [(5)](#equation-sprob4) implies that $ \mathbb{E}_t [c_{t+1}] > c_t $ (consumption is said to be a ‘submartingale’), so that consumption stochastically diverges to $ +\infty $.  The consumer’s savings also diverge to $ +\infty $.

<p><a id=fod href=#fod-link><strong>[2]</strong></a> An optimal decision rule is a map from the current state into current actions—in this case, consumption.

<p><a id=f4 href=#f4-link><strong>[3]</strong></a> Representation [(3)](#equation-sprob15ab) implies that $ d(L) = U (I - A L)^{-1} C $.

<p><a id=fn-as href=#fn-as-link><strong>[4]</strong></a> This would be the case if, for example, the [spectral radius](https://python-programming.quantecon.org/linear_algebra.html#la-neumann-remarks) of $ A $ is strictly less than one.

<p><a id=f5 href=#f5-link><strong>[5]</strong></a> A moving average representation for a process $ y_t $ is said to be **fundamental** if the linear space spanned by $ y^t $ is equal to the linear space spanned by $ w^t $.  A time-invariant innovations representation, attained via the Kalman filter, is by construction fundamental.

<p><a id=f8 href=#f8-link><strong>[6]</strong></a> See [[JYC88]](https://python-programming.quantecon.org/zreferences.html#campbellshiller88), [[LL01]](https://python-programming.quantecon.org/zreferences.html#lettlud2001), [[LL04]](https://python-programming.quantecon.org/zreferences.html#lettlud2004) for interesting applications of related ideas.
