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


<a id='optgrowth'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>


# Optimal Growth II: Accelerating the Code with Numba


## Contents

- [Optimal Growth II: Accelerating the Code with Numba](#Optimal-Growth-II:-Accelerating-the-Code-with-Numba)  
  - [Overview](#Overview)  
  - [The Model](#The-Model)  
  - [Computation](#Computation)  
  - [Exercises](#Exercises)  
  - [Solutions](#Solutions)  


In addition to what’s in Anaconda, this lecture will need the following libraries:

```python3 hide-output=true
!pip install --upgrade quantecon
!pip install --upgrade interpolation
```

## Overview

In a [previous lecture](https://python-programming.quantecon.org/optgrowth.html), we studied a stochastic optimal
growth model with one representative agent.

We solved the model using dynamic programming.

In writing our code, we focused on clarity and flexibility.

These are important, but there’s often a trade-off between flexibility and
speed.

The reason is that, when code is less flexible, we can exploit structure more
easily.

(This is true about algorithms and mathematical problems more generally:
more specific problems have more structure, which, with some thought, can be
exploited for better results.)

So, in this lecture, we are going to accept less flexibility while gaining
speed, using just-in-time (JIT) compilation to
accelerate our code.

Let’s start with some imports:

```python3 hide-output=false
import numpy as np
import matplotlib.pyplot as plt
from interpolation import interp
from numba import jit, njit, jitclass, prange, float64, int32
from quantecon.optimize.scalar_maximization import brent_max

%matplotlib inline
```

We are using an interpolation function from
[interpolation.py](https://github.com/EconForge/interpolation.py) because it
helps us JIT-compile our code.

The function brent_max is also designed for embedding in JIT-compiled code.

These are alternatives to similar functions in SciPy (which, unfortunately, are not JIT-aware).

<!-- #region -->
## The Model


<a id='index-1'></a>
The model is the same as discussed in [this lecture](https://python-programming.quantecon.org/optgrowth.html).

We will use the same algorithm to solve it—the only difference is in the
implementation itself.

We will use the CRRA utility specification

$$
u(c) = \frac{c^{1 - γ} - 1} {1 - γ}
$$

We continue to assume that

- $ f(k) = k^{\alpha} $  
- $ \phi $ is the distribution of $ \exp(\mu + s \zeta) $ when $ \zeta $ is standard normal  
<!-- #endregion -->

<!-- #region -->
## Computation


<a id='index-2'></a>
As before, we will store the primitives of the optimal growth model in a class.

But now we are going to use [Numba’s](https://python-programming.quantecon.org/numba.html) `@jitclass` decorator to
target our class for JIT compilation.

Because we are going to use Numba to compile our class, we need to specify the
types of the data:
<!-- #endregion -->

```python3 hide-output=false
opt_growth_data = [
    ('α', float64),          # Production parameter
    ('β', float64),          # Discount factor
    ('μ', float64),          # Shock location parameter
    ('γ', float64),          # Preference parameter
    ('s', float64),          # Shock scale parameter
    ('grid', float64[:]),    # Grid (array)
    ('shocks', float64[:])   # Shock draws (array)
]
```

Note the convention for specifying the types of each argument.

Now we’re ready to create our class, which will combine parameters and a
method that realizes the right hand side of the Bellman equation [(9)](https://python-programming.quantecon.org/optgrowth.html#equation-fpb30).

You will notice that, unlike in the [previous lecture](https://python-programming.quantecon.org/optgrowth.html), we
hardwire the Cobb-Douglas production and CRRA utility specifications into the
class.

Thus, we are losing flexibility, but we will gain substantial speed.

```python3 hide-output=false
@jitclass(opt_growth_data)
class OptimalGrowthModel:

    def __init__(self,
                 α=0.4,
                 β=0.96,
                 μ=0,
                 s=0.1,
                 γ=1.5,
                 grid_max=4,
                 grid_size=120,
                 shock_size=250,
                 seed=1234):

        self.α, self.β, self.γ, self.μ, self.s = α, β, γ, μ, s

        # Set up grid
        self.grid = np.linspace(1e-5, grid_max, grid_size)

        # Store shocks (with a seed, so results are reproducible)
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))

    def f(self, k):
        return k**self.α

    def u(self, c):
        return (c**(1 - self.γ) - 1) / (1 - self.γ)

    def objective(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, shocks = self.u, self.f, self.β, self.shocks

        v = lambda x: interp(self.grid, v_array, x)

        return u(c) + β * np.mean(v(f(y - c) * shocks))
```

### The Bellman Operator

Here’s a function that uses JIT compilation to accelerate the Bellman operator

```python3 hide-output=false
@jit(nopython=True)
def T(og, v):
    """
    The Bellman operator.

      * og is an instance of OptimalGrowthModel
      * v is an array representing a guess of the value function
    """
    v_new = np.empty_like(v)

    for i in range(len(og.grid)):
        y = og.grid[i]

        # Maximize RHS of Bellman equation at state y
        v_max = brent_max(og.objective, 1e-10, y, args=(y, v))[1]
        v_new[i] = v_max

    return v_new
```

Here’s another function, very similar to the last, that computes a $ v $-greedy
policy:

```python3 hide-output=false
@jit(nopython=True)
def get_greedy(og, v):
    """
    Compute a v-greedy policy.

      * og is an instance of OptimalGrowthModel
      * v is an array representing a guess of the value function
    """
    v_greedy = np.empty_like(v)

    for i in range(len(og.grid)):
        y = og.grid[i]

        # Find maximizer of RHS of Bellman equation at state y
        c_star = brent_max(og.objective, 1e-10, y, args=(y, v))[0]
        v_greedy[i] = c_star

    return v_greedy
```

The last two functions could be merged, as they were in our [previous implementation](https://python-programming.quantecon.org/optgrowth.html), but we resisted doing so to increase efficiency.

Here’s a function that iterates from a starting guess of the value function until the difference between successive iterates is below a particular tolerance level.

```python3 hide-output=false
def solve_model(og,
                tol=1e-4,
                max_iter=1000,
                verbose=True,
                print_skip=25):

    # Set up loop
    v = np.log(og.grid)  # Initial condition
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(og, v)
        error = np.max(np.abs(v - v_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        v = v_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return v_new
```

Let’s compute the approximate solution at the default parameters.

First we create an instance:

```python3 hide-output=false
og = OptimalGrowthModel()
```

Now we call `solve_model`, using the `%%time` magic to check how long it
takes.

```python3 hide-output=false
%%time
v_solution = solve_model(og)
```

You will notice that this is *much* faster than our [original implementation](https://python-programming.quantecon.org/optgrowth.html#ogex1).

Let’s plot the resulting policy:

```python3 hide-output=false
v_greedy = get_greedy(og, v_solution)

fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2,
        alpha=0.6, label='Approximate value function')

ax.legend(loc='lower right')
plt.show()
```

Everything seems in order, so our code acceleration has been successful!


## Exercises

<!-- #region -->
### Exercise 1

Once an optimal consumption policy $ \sigma $ is given, income follows

$$
y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1}
$$

The next figure shows a simulation of 100 elements of this sequence for three
different discount factors (and hence three different policies).

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/optgrowth/solution_og_ex2.png" style="">

  
In each sequence, the initial condition is $ y_0 = 0.1 $.

The discount factors are `discount_factors = (0.8, 0.9, 0.98)`.

We have also dialed down the shocks a bit with `s = 0.05`.

Otherwise, the parameters and primitives are the same as the log-linear model discussed earlier in the lecture.

Notice that more patient agents typically have higher wealth.

Replicate the figure modulo randomness.
<!-- #endregion -->

## Solutions


### Exercise 1

Here’s one solution

```python3 hide-output=false
def simulate_og(σ_func, og, y0=0.1, ts_length=100):
    '''
    Compute a time series given consumption policy σ.
    '''
    y = np.empty(ts_length)
    ξ = np.random.randn(ts_length-1)
    y[0] = y0
    for t in range(ts_length-1):
        y[t+1] = (y[t] - σ_func(y[t]))**og.α * np.exp(og.μ + og.s * ξ[t])
    return y
```

```python3 hide-output=false
fig, ax = plt.subplots()

for β in (0.8, 0.9, 0.98):

    og = OptimalGrowthModel(β=β, s=0.05)

    v_solution = solve_model(og)
    v_greedy = get_greedy(og, v_solution)

    # Define an optimal policy function
    σ_func = lambda x: interp(og.grid, v_greedy, x)
    y = simulate_og(σ_func, og)
    ax.plot(y, lw=2, alpha=0.6, label=rf'$\beta = {β}$')

ax.legend(loc='lower right')
plt.show()
```
