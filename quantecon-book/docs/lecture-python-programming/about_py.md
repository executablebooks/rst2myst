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

<!-- #region -->

<a id='about-py'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>


<a id='index-0'></a>
<!-- #endregion -->

# About Python


## Contents

- [About Python](#About-Python)  
  - [Overview](#Overview)  
  - [What’s Python?](#What’s-Python?)  
  - [Scientific Programming](#Scientific-Programming)  
  - [Learn More](#Learn-More)  


> “Python has gotten sufficiently weapons grade that we don’t descend into R
> anymore. Sorry, R people. I used to be one of you but we no longer descend
> into R.” – Chris Wiggins

<!-- #region -->
## Overview

In this lecture we will

- outline what Python is  
- showcase some of its abilities  
- compare it to some other languages.  


At this stage, it’s **not** our intention that you try to replicate all you see.

We will work through what follows at a slow pace later in the lecture series.

Our only objective for this lecture is to give you some feel of what Python is, and what it can do.
<!-- #endregion -->

## What’s Python?

[Python](https://www.python.org) is a general-purpose programming language conceived in 1989 by Dutch programmer [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum).

Python is free and open source, with development coordinated through the [Python Software Foundation](https://www.python.org/psf/).

Python has experienced rapid adoption in the last decade and is now one of the most popular programming languages.

<!-- #region -->
### Common Uses

Python is a general-purpose language used in almost all application domains such as

- communications  
- web development  
- CGI and graphical user interfaces  
- game development  
- multimedia, data processing, security, etc., etc., etc.  


Used extensively by Internet services and high tech companies including

- [Google](https://www.google.com/)  
- [Dropbox](https://www.dropbox.com/)  
- [Reddit](https://www.reddit.com/)  
- [YouTube](https://www.youtube.com/)  
- [Walt Disney Animation](https://pydanny-event-notes.readthedocs.org/en/latest/socalpiggies/20110526-wda.html).  


Python is very beginner-friendly and is often used to [teach computer science and programming](http://cacm.acm.org/blogs/blog-cacm/176450-python-is-now-the-most-popular-introductory-teaching-language-at-top-us-universities/fulltext).

For reasons we will discuss, Python is particularly popular within the scientific community with users including NASA, CERN and practically all branches of academia.

It is also [replacing familiar tools like Excel](https://news.efinancialcareers.com/us-en/3002556/python-replaced-excel-banking) in the fields of finance and banking.
<!-- #endregion -->

<!-- #region -->
### Relative Popularity

The following chart, produced using Stack Overflow Trends, shows one measure of the relative popularity of Python

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/about_py/python_vs_matlab.png" style="width:55%;height:55%">

  
The figure indicates not only that Python is widely used but also that adoption of Python has accelerated significantly since 2012.

We suspect this is driven at least in part by uptake in the scientific
domain, particularly in rapidly growing fields like data science.

For example, the popularity of [pandas](http://pandas.pydata.org/), a library for data analysis with Python has exploded, as seen here.

(The corresponding time path for MATLAB is shown for comparison)

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/about_py/pandas_vs_matlab.png" style="width:55%;height:55%">

  
Note that pandas takes off in 2012, which is the same year that we see
Python’s popularity begin to spike in the first figure.

Overall, it’s clear that

- Python is [one of the most popular programming languages worldwide](https://spectrum.ieee.org/computing/software/the-top-programming-languages-2019).  
- Python is a major tool for scientific computing, accounting for a rapidly rising share of scientific work around the globe.  
<!-- #endregion -->

### Features

Python is a [high-level language](https://en.wikipedia.org/wiki/High-level_programming_language) suitable for rapid development.

It has a relatively small core language supported by many libraries.

Other features of Python:

- multiple programming styles are supported (procedural, object-oriented, functional, etc.)  
- it is interpreted rather than compiled.  

<!-- #region -->
### Syntax and Design


<a id='index-2'></a>
One nice feature of Python is its elegant syntax — we’ll see many examples later on.

Elegant code might sound superfluous but in fact it’s highly beneficial because it makes the syntax easy to read and easy to remember.

Remembering how to read from files, sort dictionaries and other such routine tasks means that you don’t need to break your flow in order to hunt down correct syntax.

Closely related to elegant syntax is an elegant design.

Features like iterators, generators, decorators and list comprehensions make Python highly expressive, allowing you to get more done with less code.

[Namespaces](https://en.wikipedia.org/wiki/Namespace) improve productivity by cutting down on bugs and syntax errors.
<!-- #endregion -->

<!-- #region -->
## Scientific Programming


<a id='index-3'></a>
Python has become one of the core languages of scientific computing.

It’s either the dominant player or a major player in

- [machine learning and data science](http://scikit-learn.org/stable/)  
- [astronomy](http://www.astropy.org/)  
- [artificial intelligence](https://wiki.python.org/moin/PythonForArtificialIntelligence)  
- [chemistry](http://chemlab.github.io/chemlab/)  
- [computational biology](http://biopython.org/wiki/Main_Page)  
- [meteorology](https://pypi.org/project/meteorology/)  


Its popularity in economics is also beginning to rise.

This section briefly showcases some examples of Python for scientific programming.

- All of these topics will be covered in detail later on.  
<!-- #endregion -->

<!-- #region -->
### Numerical Programming


<a id='index-4'></a>
Fundamental matrix and array processing capabilities are provided by the excellent [NumPy](http://www.numpy.org/) library.

NumPy provides the basic array data type plus some simple processing operations.

For example, let’s build some arrays
<!-- #endregion -->

```python3 hide-output=false
import numpy as np                     # Load the library

a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
b = np.cos(a)                          # Apply cosine to each element of a
c = np.sin(a)                          # Apply sin to each element of a
```

Now let’s take the inner product

```python3 hide-output=false
b @ c
```

<!-- #region -->
The number you see here might vary slightly but it’s essentially zero.

(For older versions of Python and NumPy you need to use the [np.dot](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) function)

The [SciPy](http://www.scipy.org) library is built on top of NumPy and provides additional functionality.


<a id='tuple-unpacking-example'></a>
For example, let’s calculate $ \int_{-2}^2 \phi(z) dz $ where $ \phi $ is the standard normal density.
<!-- #endregion -->

```python3 hide-output=false
from scipy.stats import norm
from scipy.integrate import quad

ϕ = norm()
value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
value
```

<!-- #region -->
SciPy includes many of the standard routines used in

- [linear algebra](http://docs.scipy.org/doc/scipy/reference/linalg.html)  
- [integration](http://docs.scipy.org/doc/scipy/reference/integrate.html)  
- [interpolation](http://docs.scipy.org/doc/scipy/reference/interpolate.html)  
- [optimization](http://docs.scipy.org/doc/scipy/reference/optimize.html)  
- [distributions and random number generation](http://docs.scipy.org/doc/scipy/reference/stats.html)  
- [signal processing](http://docs.scipy.org/doc/scipy/reference/signal.html)  


See them all [here](http://docs.scipy.org/doc/scipy/reference/index.html).
<!-- #endregion -->

<!-- #region -->
### Graphics


<a id='index-5'></a>
The most popular and comprehensive Python library for creating figures and graphs is [Matplotlib](http://matplotlib.org/), with functionality including

- plots, histograms, contour images, 3D graphs, bar charts etc.  
- output in many formats (PDF, PNG, EPS, etc.)  
- LaTeX integration  


Example 2D plot with embedded LaTeX annotations

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/about_py/qs.png" style="width:55%;height:55%">

  
Example contour plot

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/about_py/bn_density1.png" style="width:40%;height:40%">

  
Example 3D plot

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/about_py/career_vf.png" style="width:50%;height:50%">

  
More examples can be found in the [Matplotlib thumbnail gallery](http://matplotlib.org/gallery.html).

Other graphics libraries include

- [Plotly](https://plot.ly/python/)  
- [Bokeh](http://bokeh.pydata.org/en/latest/)  
- [VPython](http://www.vpython.org/) — 3D graphics and animations  
<!-- #endregion -->

<!-- #region -->
### Symbolic Algebra

It’s useful to be able to manipulate symbolic expressions, as in Mathematica or Maple.


<a id='index-6'></a>
The [SymPy](http://www.sympy.org/) library provides this functionality from within the Python shell.
<!-- #endregion -->

```python3 hide-output=false
from sympy import Symbol

x, y = Symbol('x'), Symbol('y')  # Treat 'x' and 'y' as algebraic symbols
x + x + x + y
```

We can manipulate expressions

```python3 hide-output=false
expression = (x + y)**2
expression.expand()
```

solve polynomials

```python3 hide-output=false
from sympy import solve

solve(x**2 + x + 2)
```

and calculate limits, derivatives and integrals

```python3 hide-output=false
from sympy import limit, sin, diff

limit(1 / x, x, 0)
```

```python3 hide-output=false
limit(sin(x) / x, x, 0)
```

```python3 hide-output=false
diff(sin(x), x)
```

The beauty of importing this functionality into Python is that we are working within
a fully fledged programming language.

We can easily create tables of derivatives, generate LaTeX output, add that output
to figures and so on.


### Statistics

Python’s data manipulation and statistics libraries have improved rapidly over
the last few years.

<!-- #region -->
#### Pandas


<a id='index-7'></a>
One of the most popular libraries for working with data is [pandas](http://pandas.pydata.org/).

Pandas is fast, efficient, flexible and well designed.

Here’s a simple example, using some dummy data generated with Numpy’s excellent
`random` functionality.
<!-- #endregion -->

```python3 hide-output=false
import pandas as pd
np.random.seed(1234)

data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
dates = pd.date_range('28/12/2010', periods=5)

df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
print(df)
```

```python3 hide-output=false
df.mean()
```

<!-- #region -->
#### Other Useful Statistics Libraries


<a id='index-8'></a>
- [statsmodels](http://statsmodels.sourceforge.net/) — various statistical routines  



<a id='index-9'></a>
- [scikit-learn](http://scikit-learn.org/) — machine learning in Python (sponsored by Google, among others)  



<a id='index-10'></a>
- [pyMC](http://pymc-devs.github.io/pymc/) — for Bayesian data analysis  



<a id='index-11'></a>
- [pystan](https://pystan.readthedocs.org/en/latest/) Bayesian analysis based on [stan](http://mc-stan.org/)  
<!-- #endregion -->

<!-- #region -->
### Networks and Graphs

Python has many libraries for studying graphs.


<a id='index-12'></a>
One well-known example is [NetworkX](http://networkx.github.io/).
Its features include, among many other things:

- standard graph algorithms for analyzing networks  
- plotting routines  


Here’s some example code that generates and plots a random graph, with node color determined by shortest path length from a central node.
<!-- #endregion -->

```python3 hide-output=false
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(1234)

# Generate a random graph
p = dict((i, (np.random.uniform(0, 1), np.random.uniform(0, 1)))
         for i in range(200))
g = nx.random_geometric_graph(200, 0.12, pos=p)
pos = nx.get_node_attributes(g, 'pos')

# Find node nearest the center point (0.5, 0.5)
dists = [(x - 0.5)**2 + (y - 0.5)**2 for x, y in list(pos.values())]
ncenter = np.argmin(dists)

# Plot graph, coloring by path length from central node
p = nx.single_source_shortest_path_length(g, ncenter)
plt.figure()
nx.draw_networkx_edges(g, pos, alpha=0.4)
nx.draw_networkx_nodes(g,
                       pos,
                       nodelist=list(p.keys()),
                       node_size=120, alpha=0.5,
                       node_color=list(p.values()),
                       cmap=plt.cm.jet_r)
plt.show()
```

<!-- #region -->
### Cloud Computing


<a id='index-13'></a>
Running your Python code on massive servers in the cloud is becoming easier and easier.


<a id='index-14'></a>
A nice example is [Anaconda Enterprise](https://www.anaconda.com/enterprise/).

See also


<a id='index-15'></a>
- [Amazon Elastic Compute Cloud](http://aws.amazon.com/ec2/)  



<a id='index-16'></a>
- The [Google App Engine](https://cloud.google.com/appengine/) (Python, Java, PHP or Go)  



<a id='index-17'></a>
- [Pythonanywhere](https://www.pythonanywhere.com/)  



<a id='index-18'></a>
- [Sagemath Cloud](https://cloud.sagemath.com/)  
<!-- #endregion -->

<!-- #region -->
### Parallel Processing


<a id='index-19'></a>
Apart from the cloud computing options listed above, you might like to consider


<a id='index-20'></a>
- [Parallel computing through IPython clusters](http://ipython.org/ipython-doc/stable/parallel/parallel_demos.html).  



<a id='index-21'></a>
- The [Starcluster](http://star.mit.edu/cluster/) interface to Amazon’s EC2.  



<a id='index-23'></a>
- GPU programming through [PyCuda](https://wiki.tiker.net/PyCuda), [PyOpenCL](https://mathema.tician.de/software/pyopencl/), [Theano](http://deeplearning.net/software/theano/) or similar.  



<a id='intfc'></a>
<!-- #endregion -->

<!-- #region -->
### Other Developments

There are many other interesting developments with scientific programming in Python.

Some representative examples include


<a id='index-24'></a>
- [Jupyter](http://jupyter.org/) — Python in your browser with interactive code cells,  embedded images and other useful features.  



<a id='index-25'></a>
- [Numba](http://numba.pydata.org/) — Make Python run at the same speed as native machine code!  



<a id='index-26'></a>
- [Blaze](http://blaze.pydata.org/) — a generalization of NumPy.  



<a id='index-27'></a>
- [PyTables](http://www.pytables.org) — manage large data sets.  



<a id='index-28'></a>
- [CVXPY](https://github.com/cvxgrp/cvxpy) — convex optimization in Python.  
<!-- #endregion -->

<!-- #region -->
## Learn More

- Browse some Python projects on [GitHub](https://github.com/trending?l=python).  
- Read more about [Python’s history and rise in popularity](https://www.welcometothejungle.com/en/articles/btc-python-popular) .  
- Have a look at [some of the Jupyter notebooks](http://nbviewer.jupyter.org/) people have shared on various scientific topics.  



<a id='index-29'></a>
- Visit the [Python Package Index](https://pypi.org/).  
- View some of the questions people are asking about Python on [Stackoverflow](http://stackoverflow.com/questions/tagged/python).  
- Keep up to date on what’s happening in the Python community with the [Python subreddit](https://www.reddit.com:443/r/Python/).  
<!-- #endregion -->
