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


<a id='python-done-right'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>


# Python Essentials


## Contents

- [Python Essentials](#Python-Essentials)  
  - [Overview](#Overview)  
  - [Data Types](#Data-Types)  
  - [Input and Output](#Input-and-Output)  
  - [Iterating](#Iterating)  
  - [Comparisons and Logical Operators](#Comparisons-and-Logical-Operators)  
  - [More Functions](#More-Functions)  
  - [Coding Style and PEP8](#Coding-Style-and-PEP8)  
  - [Exercises](#Exercises)  
  - [Solutions](#Solutions)  


## Overview

We have covered a lot of material quite quickly, with a focus on examples.

Now let’s cover some core features of Python in a more systematic way.

This approach is less exciting but helps clear up some details.

<!-- #region -->
## Data Types


<a id='index-0'></a>
Computer programs typically keep track of a range of data types.

For example, `1.5` is a floating point number, while `1` is an integer.

Programs need to distinguish between these two types for various reasons.

One is that they are stored in memory differently.

Another is that arithmetic operations are different

- For example, floating point arithmetic is implemented on most machines by a
  specialized Floating Point Unit (FPU).  


In general, floats are more informative but arithmetic operations on integers
are faster and more accurate.

Python provides numerous other built-in Python data types, some of which we’ve already met

- strings, lists, etc.  


Let’s learn a bit more about them.
<!-- #endregion -->

### Primitive Data Types

One simple data type is **Boolean values**, which can be either `True` or `False`

```python3 hide-output=false
x = True
x
```

We can check the type of any object in memory using the `type()` function.

```python3 hide-output=false
type(x)
```

In the next line of code, the interpreter evaluates the expression on the right of = and binds y to this value

```python3 hide-output=false
y = 100 < 10
y
```

```python3 hide-output=false
type(y)
```

In arithmetic expressions, `True` is converted to `1` and `False` is converted `0`.

This is called **Boolean arithmetic** and is often useful in programming.

Here are some examples

```python3 hide-output=false
x + y
```

```python3 hide-output=false
x * y
```

```python3 hide-output=false
True + True
```

```python3 hide-output=false
bools = [True, True, False, True]  # List of Boolean values

sum(bools)
```

Complex numbers are another primitive data type in Python

```python3 hide-output=false
x = complex(1, 2)
y = complex(2, 1)
print(x * y)

type(x)
```

<!-- #region -->
### Containers

Python has several basic types for storing collections of (possibly heterogeneous) data.

We’ve [already discussed lists](https://python-programming.quantecon.org/python_by_example.html#lists-ref).


<a id='index-1'></a>
A related data type is **tuples**, which are “immutable” lists
<!-- #endregion -->

```python3 hide-output=false
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
```

```python3 hide-output=false
type(x)
```

In Python, an object is called **immutable** if, once created, the object cannot be changed.

Conversely, an object is **mutable** if it can still be altered after creation.

Python lists are mutable

```python3 hide-output=false
x = [1, 2]
x[0] = 10
x
```

But tuples are not

```python3 hide-output=false
x = (1, 2)
x[0] = 10
```

We’ll say more about the role of mutable and immutable data a bit later.

Tuples (and lists) can be “unpacked” as follows

```python3 hide-output=false
integers = (10, 20, 30)
x, y, z = integers
x
```

```python3 hide-output=false
y
```

You’ve actually [seen an example of this](https://python-programming.quantecon.org/about_py.html#tuple-unpacking-example) already.

Tuple unpacking is convenient and we’ll use it often.

<!-- #region -->
#### Slice Notation


<a id='index-2'></a>
To access multiple elements of a list or tuple, you can use Python’s slice
notation.

For example,
<!-- #endregion -->

```python3 hide-output=false
a = [2, 4, 6, 8]
a[1:]
```

```python3 hide-output=false
a[1:3]
```

The general rule is that `a[m:n]` returns `n - m` elements, starting at `a[m]`.

Negative numbers are also permissible

```python3 hide-output=false
a[-2:]  # Last two elements of the list
```

The same slice notation works on tuples and strings

```python3 hide-output=false
s = 'foobar'
s[-3:]  # Select the last three elements
```

<!-- #region -->
#### Sets and Dictionaries


<a id='index-4'></a>
Two other container types we should mention before moving on are [sets](https://docs.python.org/3/tutorial/datastructures.html#sets) and [dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).

Dictionaries are much like lists, except that the items are named instead of
numbered
<!-- #endregion -->

```python3 hide-output=false
d = {'name': 'Frodo', 'age': 33}
type(d)
```

```python3 hide-output=false
d['age']
```

The names `'name'` and `'age'` are called the *keys*.

The objects that the keys are mapped to (`'Frodo'` and `33`) are called the `values`.

Sets are unordered collections without duplicates, and set methods provide the
usual set-theoretic operations

```python3 hide-output=false
s1 = {'a', 'b'}
type(s1)
```

```python3 hide-output=false
s2 = {'b', 'c'}
s1.issubset(s2)
```

```python3 hide-output=false
s1.intersection(s2)
```

The `set()` function creates sets from sequences

```python3 hide-output=false
s3 = set(('foo', 'bar', 'foo'))
s3
```

<!-- #region -->
## Input and Output


<a id='index-5'></a>
Let’s briefly review reading and writing to text files, starting with writing
<!-- #endregion -->

```python3 hide-output=false
f = open('newfile.txt', 'w')   # Open 'newfile.txt' for writing
f.write('Testing\n')           # Here '\n' means new line
f.write('Testing again')
f.close()
```

<!-- #region -->
Here

- The built-in function `open()` creates a file object for writing to.  
- Both `write()` and `close()` are methods of file objects.  


Where is this file that we’ve created?

Recall that Python maintains a concept of the present working directory (pwd) that can be located from with Jupyter or IPython via
<!-- #endregion -->

```python3 hide-output=false
%pwd
```

If a path is not specified, then this is where Python writes to.

We can also use Python to read the contents of `newline.txt` as follows

```python3 hide-output=false
f = open('newfile.txt', 'r')
out = f.read()
out
```

```python3 hide-output=false
print(out)
```

<!-- #region -->
### Paths


<a id='index-6'></a>
Note that if `newfile.txt` is not in the present working directory then this call to `open()` fails.

In this case, you can shift the file to the pwd or specify the [full path](https://en.wikipedia.org/wiki/Path_%28computing%29) to the file
<!-- #endregion -->

<!-- #region hide-output=false -->
```python3
f = open('insert_full_path_to_file/newfile.txt', 'r')
```

<!-- #endregion -->


<a id='iterating-version-1'></a>

<!-- #region -->
## Iterating


<a id='index-7'></a>
One of the most important tasks in computing is stepping through a
sequence of data and performing a given action.

One of Python’s strengths is its simple, flexible interface to this kind of iteration via
the `for` loop.
<!-- #endregion -->

<!-- #region -->
### Looping over Different Objects

Many Python objects are “iterable”, in the sense that they can be looped over.

To give an example, let’s write the file us_cities.txt, which lists US cities and their population, to the present working directory.


<a id='us-cities-data'></a>
<!-- #endregion -->

```python3 hide-output=false
%%file us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229
```

Here %%file is an [IPython cell magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cell-magics).

Suppose that we want to make the information more readable, by capitalizing names and adding commas to mark thousands.

The program below reads the data in and makes the conversion:

```python3 hide-output=false
data_file = open('us_cities.txt', 'r')
for line in data_file:
    city, population = line.split(':')         # Tuple unpacking
    city = city.title()                        # Capitalize city names
    population = f'{int(population):,}'        # Add commas to numbers
    print(city.ljust(15) + population)
data_file.close()
```

<!-- #region -->
Here `format()` is a string method [used for inserting variables into strings](https://docs.python.org/3/library/string.html#formatspec).

The reformatting of each line is the result of three different string methods,
the details of which can be left till later.

The interesting part of this program for us is line 2, which shows that

1. The file object `data_file` is iterable, in the sense that it can be placed to the right of `in` within a `for` loop.  
1. Iteration steps through each line in the file.  


This leads to the clean, convenient syntax shown in our program.

Many other kinds of objects are iterable, and we’ll discuss some of them later on.
<!-- #endregion -->

### Looping without Indices

One thing you might have noticed is that Python tends to favor looping without explicit indexing.

For example,

```python3 hide-output=false
x_values = [1, 2, 3]  # Some iterable x
for x in x_values:
    print(x * x)
```

is preferred to

```python3 hide-output=false
for i in range(len(x_values)):
    print(x_values[i] * x_values[i])
```

When you compare these two alternatives, you can see why the first one is preferred.

Python provides some facilities to simplify looping without indices.

One is `zip()`, which is used for stepping through pairs from two sequences.

For example, try running the following code

```python3 hide-output=false
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')
```

The `zip()` function is also useful for creating dictionaries — for
example

```python3 hide-output=false
names = ['Tom', 'John']
marks = ['E', 'F']
dict(zip(names, marks))
```

If we actually need the index from a list, one option is to use `enumerate()`.

To understand what `enumerate()` does, consider the following example

```python3 hide-output=false
letter_list = ['a', 'b', 'c']
for index, letter in enumerate(letter_list):
    print(f"letter_list[{index}] = '{letter}'")
```

<!-- #region -->
### List Comprehensions


<a id='index-8'></a>
We can also simplify the code for generating the list of random draws considerably by using something called a *list comprehension*.

[List comprehensions](https://en.wikipedia.org/wiki/List_comprehension) are an elegant Python tool for creating lists.

Consider the following example, where the list comprehension is on the
right-hand side of the second line
<!-- #endregion -->

```python3 hide-output=false
animals = ['dog', 'cat', 'bird']
plurals = [animal + 's' for animal in animals]
plurals
```

Here’s another example

```python3 hide-output=false
range(8)
```

```python3 hide-output=false
doubles = [2 * x for x in range(8)]
doubles
```

## Comparisons and Logical Operators

<!-- #region -->
### Comparisons


<a id='index-9'></a>
Many different kinds of expressions evaluate to one of the Boolean values (i.e., `True` or `False`).

A common type is comparisons, such as
<!-- #endregion -->

```python3 hide-output=false
x, y = 1, 2
x < y
```

```python3 hide-output=false
x > y
```

One of the nice features of Python is that we can *chain* inequalities

```python3 hide-output=false
1 < 2 < 3
```

```python3 hide-output=false
1 <= 2 <= 3
```

As we saw earlier, when testing for equality we use `==`

```python3 hide-output=false
x = 1    # Assignment
x == 2   # Comparison
```

For “not equal” use `!=`

```python3 hide-output=false
1 != 2
```

Note that when testing conditions, we can use **any** valid Python expression

```python3 hide-output=false
x = 'yes' if 42 else 'no'
x
```

```python3 hide-output=false
x = 'yes' if [] else 'no'
x
```

What’s going on here?

The rule is:

- Expressions that evaluate to zero, empty sequences or containers (strings, lists, etc.) and `None` are all equivalent to `False`.  
  
  - for example, `[]` and `()` are equivalent to `False` in an `if` clause  
  
- All other values are equivalent to `True`.  
  
  - for example, `42` is equivalent to `True` in an `if` clause  

<!-- #region -->
### Combining Expressions


<a id='index-10'></a>
We can combine expressions using `and`, `or` and `not`.

These are the standard logical connectives (conjunction, disjunction and denial)
<!-- #endregion -->

```python3 hide-output=false
1 < 2 and 'f' in 'foo'
```

```python3 hide-output=false
1 < 2 and 'g' in 'foo'
```

```python3 hide-output=false
1 < 2 or 'g' in 'foo'
```

```python3 hide-output=false
not True
```

```python3 hide-output=false
not not True
```

Remember

- `P and Q` is `True` if both are `True`, else `False`  
- `P or Q` is `False` if both are `False`, else `True`  

<!-- #region -->
## More Functions


<a id='index-11'></a>
Let’s talk a bit more about functions, which are all important for good programming style.
<!-- #endregion -->

<!-- #region -->
### The Flexibility of Python Functions

As we discussed in the [previous lecture](https://python-programming.quantecon.org/python_by_example.html#python-by-example), Python functions are very flexible.

In particular

- Any number of functions can be defined in a given file.  
- Functions can be (and often are) defined inside other functions.  
- Any object can be passed to a function as an argument, including other functions.  
- A function can return any kind of object, including functions.  


We already [gave an example](https://python-programming.quantecon.org/functions.html#test-program-6) of how straightforward it is to pass a function to
a function.

Note that a function can have arbitrarily many `return` statements (including zero).

Execution of the function terminates when the first return is hit, allowing
code like the following example
<!-- #endregion -->

```python3 hide-output=false
def f(x):
    if x < 0:
        return 'negative'
    return 'nonnegative'
```

Functions without a return statement automatically return the special Python object `None`.

<!-- #region -->
### Docstrings


<a id='index-12'></a>
Python has a system for adding comments to functions, modules, etc. called *docstrings*.

The nice thing about docstrings is that they are available at run-time.

Try running this
<!-- #endregion -->

```python3 hide-output=false
def f(x):
    """
    This function squares its argument
    """
    return x**2
```

After running this code, the docstring is available

```python3 hide-output=false
f?
```

<!-- #region hide-output=false -->
```ipython
Type:       function
String Form:<function f at 0x2223320>
File:       /home/john/temp/temp.py
Definition: f(x)
Docstring:  This function squares its argument
```

<!-- #endregion -->

```python3 hide-output=false
f??
```

<!-- #region hide-output=false -->
```ipython
Type:       function
String Form:<function f at 0x2223320>
File:       /home/john/temp/temp.py
Definition: f(x)
Source:
def f(x):
    """
    This function squares its argument
    """
    return x**2
```

<!-- #endregion -->

With one question mark we bring up the docstring, and with two we get the source code as well.

<!-- #region -->
### One-Line Functions: `lambda`


<a id='index-13'></a>
The `lambda` keyword is used to create simple functions on one line.

For example, the definitions
<!-- #endregion -->

```python3 hide-output=false
def f(x):
    return x**3
```

and

```python3 hide-output=false
f = lambda x: x**3
```

are entirely equivalent.

To see why `lambda` is useful, suppose that we want to calculate $ \int_0^2 x^3 dx $ (and have forgotten our high-school calculus).

The SciPy library has a function called `quad` that will do this calculation for us.

The syntax of the `quad` function is `quad(f, a, b)` where `f` is a function and `a` and `b` are numbers.

To create the function $ f(x) = x^3 $ we can use `lambda` as follows

```python3 hide-output=false
from scipy.integrate import quad

quad(lambda x: x**3, 0, 2)
```

Here the function created by `lambda` is said to be *anonymous* because it was never given a name.

<!-- #region -->
### Keyword Arguments


<a id='index-14'></a>
In a [previous lecture](https://python-programming.quantecon.org/python_by_example.html#python-by-example), you came across the statement
<!-- #endregion -->

<!-- #region hide-output=false -->
```python3
plt.plot(x, 'b-', label="white noise")
```

<!-- #endregion -->

<!-- #region -->
In this call to Matplotlib’s `plot` function, notice that the last argument is passed in `name=argument` syntax.

This is called a *keyword argument*, with `label` being the keyword.

Non-keyword arguments are called *positional arguments*, since their meaning
is determined by order

- `plot(x, 'b-', label="white noise")` is different from `plot('b-', x, label="white noise")`  


Keyword arguments are particularly useful when a function has a lot of arguments, in which case it’s hard to remember the right order.

You can adopt keyword arguments in user-defined functions with no difficulty.

The next example illustrates the syntax
<!-- #endregion -->

```python3 hide-output=false
def f(x, a=1, b=1):
    return a + b * x
```

The keyword argument values we supplied in the definition of `f` become the default values

```python3 hide-output=false
f(2)
```

They can be modified as follows

```python3 hide-output=false
f(2, a=4, b=5)
```

<!-- #region -->
## Coding Style and PEP8


<a id='index-15'></a>
To learn more about the Python programming philosophy type `import this` at the prompt.

Among other things, Python strongly favors consistency in programming style.

We’ve all heard the saying about consistency and little minds.

In programming, as in mathematics, the opposite is true

- A mathematical paper where the symbols $ \cup $ and $ \cap $ were
  reversed would be very hard to read, even if the author told you so on the
  first page.  


In Python, the standard style is set out in [PEP8](https://www.python.org/dev/peps/pep-0008/).

(Occasionally we’ll deviate from PEP8 in these lectures to better match mathematical notation)
<!-- #endregion -->

<!-- #region -->
## Exercises

Solve the following exercises.

(For some, the built-in function `sum()` comes in handy).


<a id='pyess-ex1'></a>
<!-- #endregion -->

<!-- #region -->
### Exercise 1

Part 1: Given two numeric lists or tuples `x_vals` and `y_vals` of equal length, compute
their inner product using `zip()`.

Part 2: In one line, count the number of even numbers in 0,…,99.

- Hint: `x % 2` returns 0 if `x` is even, 1 otherwise.  


Part 3: Given `pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`, count the number of pairs `(a, b)`
such that both `a` and `b` are even.


<a id='pyess-ex2'></a>
<!-- #endregion -->

<!-- #region -->
### Exercise 2

Consider the polynomial


<a id='equation-polynom0'></a>
$$
p(x)
= a_0 + a_1 x + a_2 x^2 + \cdots a_n x^n
= \sum_{i=0}^n a_i x^i \tag{1}
$$

Write a function `p` such that `p(x, coeff)` that computes the value in [(1)](#equation-polynom0) given a point `x` and a list of coefficients `coeff`.

Try to use `enumerate()` in your loop.


<a id='pyess-ex3'></a>
<!-- #endregion -->

<!-- #region -->
### Exercise 3

Write a function that takes a string as an argument and returns the number of capital letters in the string.

Hint: `'foo'.upper()` returns `'FOO'`.


<a id='pyess-ex4'></a>
<!-- #endregion -->

<!-- #region -->
### Exercise 4

Write a function that takes two sequences `seq_a` and `seq_b` as arguments and
returns `True` if every element in `seq_a` is also an element of `seq_b`, else
`False`.

- By “sequence” we mean a list, a tuple or a string.  
- Do the exercise without using [sets](https://docs.python.org/3/tutorial/datastructures.html#sets) and set methods.  



<a id='pyess-ex5'></a>
<!-- #endregion -->

<!-- #region -->
### Exercise 5

When we cover the numerical libraries, we will see they include many
alternatives for interpolation and function approximation.

Nevertheless, let’s write our own function approximation routine as an exercise.

In particular, without using any imports, write a function `linapprox` that takes as arguments

- A function `f` mapping some interval $ [a, b] $ into $ \mathbb R $.  
- Two scalars `a` and `b` providing the limits of this interval.  
- An integer `n` determining the number of grid points.  
- A number `x` satisfying `a <= x <= b`.  


and returns the [piecewise linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation) of `f` at `x`, based on `n` evenly spaced grid points `a = point[0] < point[1] < ... < point[n-1] = b`.

Aim for clarity, not efficiency.
<!-- #endregion -->

### Exercise 6

Using list comprehension syntax, we can simplify the loop in the following
code.

```python3 hide-output=false
import numpy as np

n = 100
ϵ_values = []
for i in range(n):
    e = np.random.randn()
    ϵ_values.append(e)
```

## Solutions


### Exercise 1


#### Part 1 Solution:

Here’s one possible solution

```python3 hide-output=false
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
```

This also works

```python3 hide-output=false
sum(x * y for x, y in zip(x_vals, y_vals))
```

#### Part 2 Solution:

One solution is

```python3 hide-output=false
sum([x % 2 == 0 for x in range(100)])
```

This also works:

```python3 hide-output=false
sum(x % 2 == 0 for x in range(100))
```

Some less natural alternatives that nonetheless help to illustrate the
flexibility of list comprehensions are

```python3 hide-output=false
len([x for x in range(100) if x % 2 == 0])
```

and

```python3 hide-output=false
sum([1 for x in range(100) if x % 2 == 0])
```

#### Part 3 Solution

Here’s one possibility

```python3 hide-output=false
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])
```

### Exercise 2

```python3 hide-output=false
def p(x, coeff):
    return sum(a * x**i for i, a in enumerate(coeff))
```

```python3 hide-output=false
p(1, (2, 4))
```

### Exercise 3

Here’s one solution:

```python3 hide-output=false
def f(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count

f('The Rain in Spain')
```

An alternative, more pythonic solution:

```python3 hide-output=false
def count_uppercase_chars(s):
    return sum([c.isupper() for c in s])

count_uppercase_chars('The Rain in Spain')
```

### Exercise 4

Here’s a solution:

```python3 hide-output=false
def f(seq_a, seq_b):
    is_subset = True
    for a in seq_a:
        if a not in seq_b:
            is_subset = False
    return is_subset

# == test == #

print(f([1, 2], [1, 2, 3]))
print(f([1, 2, 3], [1, 2]))
```

Of course, if we use the `sets` data type then the solution is easier

```python3 hide-output=false
def f(seq_a, seq_b):
    return set(seq_a).issubset(set(seq_b))
```

### Exercise 5

```python3 hide-output=false
def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval
    [a, b], with n evenly spaced grid points.

    Parameters
    ==========
        f : function
            The function to approximate

        x, a, b : scalars (floats or integers)
            Evaluation point and endpoints, with a <= x <= b

        n : integer
            Number of grid points

    Returns
    =======
        A float. The interpolant evaluated at x

    """
    length_of_interval = b - a
    num_subintervals = n - 1
    step = length_of_interval / num_subintervals

    # === find first grid point larger than x === #
    point = a
    while point <= x:
        point += step

    # === x must lie between the gridpoints (point - step) and point === #
    u, v = point - step, point

    return f(u) + (x - u) * (f(v) - f(u)) / (v - u)
```

### Exercise 6

Here’s one solution.

```python3 hide-output=false
n = 100
ϵ_values = [np.random.randn() for i in range(n)]
```
