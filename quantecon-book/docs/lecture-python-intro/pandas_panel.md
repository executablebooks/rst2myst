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


<a id='ppd'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>

<!-- #region -->
# Pandas for Panel Data


<a id='index-1'></a>
<!-- #endregion -->

## Contents

- [Pandas for Panel Data](#Pandas-for-Panel-Data)  
  - [Overview](#Overview)  
  - [Slicing and Reshaping Data](#Slicing-and-Reshaping-Data)  
  - [Merging Dataframes and Filling NaNs](#Merging-Dataframes-and-Filling-NaNs)  
  - [Grouping and Summarizing Data](#Grouping-and-Summarizing-Data)  
  - [Final Remarks](#Final-Remarks)  
  - [Exercises](#Exercises)  
  - [Solutions](#Solutions)  

<!-- #region -->
## Overview

In an [earlier lecture on pandas](https://python-programming.quantecon.org/pandas.html), we looked at working with simple data sets.

Econometricians often need to work with more complex data sets, such as panels.

Common tasks include

- Importing data, cleaning it and reshaping it across several axes.  
- Selecting a time series or cross-section from a panel.  
- Grouping and summarizing data.  


`pandas` (derived from ‘panel’ and ‘data’) contains powerful and
easy-to-use tools for solving exactly these kinds of problems.

In what follows, we will use a panel data set of real minimum wages from the OECD to create:

- summary statistics over multiple dimensions of our data  
- a time series of the average minimum wage of countries in the dataset  
- kernel density estimates of wages by continent  


We will begin by reading in our long format panel data from a CSV file and
reshaping the resulting `DataFrame` with `pivot_table` to build a `MultiIndex`.

Additional detail will be added to our `DataFrame` using pandas’
`merge` function, and data will be summarized with the `groupby`
function.
<!-- #endregion -->

## Slicing and Reshaping Data

We will read in a dataset from the OECD of real minimum wages in 32
countries and assign it to `realwage`.

The dataset can be accessed with the following link:

```python3 hide-output=false
url1 = 'https://raw.githubusercontent.com/QuantEcon/lecture-source-py/master/source/_static/lecture_specific/pandas_panel/realwage.csv'
```

```python3 hide-output=false
import pandas as pd

# Display 6 columns for viewing purposes
pd.set_option('display.max_columns', 6)

# Reduce decimal points to 2
pd.options.display.float_format = '{:,.2f}'.format

realwage = pd.read_csv(url1)
```

Let’s have a look at what we’ve got to work with

```python3 hide-output=false
realwage.head()  # Show first 5 rows
```

The data is currently in long format, which is difficult to analyze when there are several dimensions to the data.

We will use `pivot_table` to create a wide format panel, with a `MultiIndex` to handle higher dimensional data.

`pivot_table` arguments should specify the data (values), the index, and the columns we want in our resulting dataframe.

By passing a list in columns, we can create a `MultiIndex` in our column axis

```python3 hide-output=false
realwage = realwage.pivot_table(values='value',
                                index='Time',
                                columns=['Country', 'Series', 'Pay period'])
realwage.head()
```

To more easily filter our time series data, later on, we will convert the index into a `DateTimeIndex`

```python3 hide-output=false
realwage.index = pd.to_datetime(realwage.index)
type(realwage.index)
```

The columns contain multiple levels of indexing, known as a
`MultiIndex`, with levels being ordered hierarchically (Country >
Series > Pay period).

A `MultiIndex` is the simplest and most flexible way to manage panel
data in pandas

```python3 hide-output=false
type(realwage.columns)
```

```python3 hide-output=false
realwage.columns.names
```

Like before, we can select the country (the top level of our
`MultiIndex`)

```python3 hide-output=false
realwage['United States'].head()
```

Stacking and unstacking levels of the `MultiIndex` will be used
throughout this lecture to reshape our dataframe into a format we need.

`.stack()` rotates the lowest level of the column `MultiIndex` to
the row index (`.unstack()` works in the opposite direction - try it
out)

```python3 hide-output=false
realwage.stack().head()
```

We can also pass in an argument to select the level we would like to
stack

```python3 hide-output=false
realwage.stack(level='Country').head()
```

Using a `DatetimeIndex` makes it easy to select a particular time
period.

Selecting one year and stacking the two lower levels of the
`MultiIndex` creates a cross-section of our panel data

```python3 hide-output=false
realwage['2015'].stack(level=(1, 2)).transpose().head()
```

For the rest of lecture, we will work with a dataframe of the hourly
real minimum wages across countries and time, measured in 2015 US
dollars.

To create our filtered dataframe (`realwage_f`), we can use the `xs`
method to select values at lower levels in the multiindex, while keeping
the higher levels (countries in this case)

```python3 hide-output=false
realwage_f = realwage.xs(('Hourly', 'In 2015 constant prices at 2015 USD exchange rates'),
                         level=('Pay period', 'Series'), axis=1)
realwage_f.head()
```

## Merging Dataframes and Filling NaNs

Similar to relational databases like SQL, pandas has built in methods to
merge datasets together.

Using country information from
[WorldData.info](https://www.worlddata.info/downloads/), we’ll add
the continent of each country to `realwage_f` with the `merge`
function.

The dataset can be accessed with the following link:

```python3 hide-output=false
url2 = 'https://raw.githubusercontent.com/QuantEcon/lecture-source-py/master/source/_static/lecture_specific/pandas_panel/countries.csv'
```

```python3 hide-output=false
worlddata = pd.read_csv(url2, sep=';')
worlddata.head()
```

First, we’ll select just the country and continent variables from
`worlddata` and rename the column to ‘Country’

```python3 hide-output=false
worlddata = worlddata[['Country (en)', 'Continent']]
worlddata = worlddata.rename(columns={'Country (en)': 'Country'})
worlddata.head()
```

We want to merge our new dataframe, `worlddata`, with `realwage_f`.

The pandas `merge` function allows dataframes to be joined together by
rows.

Our dataframes will be merged using country names, requiring us to use
the transpose of `realwage_f` so that rows correspond to country names
in both dataframes

```python3 hide-output=false
realwage_f.transpose().head()
```

<!-- #region -->
We can use either left, right, inner, or outer join to merge our
datasets:

- left join includes only countries from the left dataset  
- right join includes only countries from the right dataset  
- outer join includes countries that are in either the left and right datasets  
- inner join includes only countries common to both the left and right datasets  


By default, `merge` will use an inner join.

Here we will pass `how='left'` to keep all countries in
`realwage_f`, but discard countries in `worlddata` that do not have
a corresponding data entry `realwage_f`.

This is illustrated by the red shading in the following diagram

<img src="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/lecture_specific/pandas_panel/venn_diag.png" style="">

  
We will also need to specify where the country name is located in each
dataframe, which will be the `key` that is used to merge the
dataframes ‘on’.

Our ‘left’ dataframe (`realwage_f.transpose()`) contains countries in
the index, so we set `left_index=True`.

Our ‘right’ dataframe (`worlddata`) contains countries in the
‘Country’ column, so we set `right_on='Country'`
<!-- #endregion -->

```python3 hide-output=false
merged = pd.merge(realwage_f.transpose(), worlddata,
                  how='left', left_index=True, right_on='Country')
merged.head()
```

Countries that appeared in `realwage_f` but not in `worlddata` will
have `NaN` in the Continent column.

To check whether this has occurred, we can use `.isnull()` on the
continent column and filter the merged dataframe

```python3 hide-output=false
merged[merged['Continent'].isnull()]
```

We have three missing values!

One option to deal with NaN values is to create a dictionary containing
these countries and their respective continents.

`.map()` will match countries in `merged['Country']` with their
continent from the dictionary.

Notice how countries not in our dictionary are mapped with `NaN`

```python3 hide-output=false
missing_continents = {'Korea': 'Asia',
                      'Russian Federation': 'Europe',
                      'Slovak Republic': 'Europe'}

merged['Country'].map(missing_continents)
```

We don’t want to overwrite the entire series with this mapping.

`.fillna()` only fills in `NaN` values in `merged['Continent']`
with the mapping, while leaving other values in the column unchanged

```python3 hide-output=false
merged['Continent'] = merged['Continent'].fillna(merged['Country'].map(missing_continents))

# Check for whether continents were correctly mapped

merged[merged['Country'] == 'Korea']
```

We will also combine the Americas into a single continent - this will make our visualization nicer later on.

To do this, we will use `.replace()` and loop through a list of the continent values we want to replace

```python3 hide-output=false
replace = ['Central America', 'North America', 'South America']

for country in replace:
    merged['Continent'].replace(to_replace=country,
                                value='America',
                                inplace=True)
```

Now that we have all the data we want in a single `DataFrame`, we will
reshape it back into panel form with a `MultiIndex`.

We should also ensure to sort the index using `.sort_index()` so that we
can efficiently filter our dataframe later on.

By default, levels will be sorted top-down

```python3 hide-output=false
merged = merged.set_index(['Continent', 'Country']).sort_index()
merged.head()
```

While merging, we lost our `DatetimeIndex`, as we merged columns that
were not in datetime format

```python3 hide-output=false
merged.columns
```

Now that we have set the merged columns as the index, we can recreate a
`DatetimeIndex` using `.to_datetime()`

```python3 hide-output=false
merged.columns = pd.to_datetime(merged.columns)
merged.columns = merged.columns.rename('Time')
merged.columns
```

The `DatetimeIndex` tends to work more smoothly in the row axis, so we
will go ahead and transpose `merged`

```python3 hide-output=false
merged = merged.transpose()
merged.head()
```

## Grouping and Summarizing Data

Grouping and summarizing data can be particularly useful for
understanding large panel datasets.

A simple way to summarize data is to call an [aggregation
method](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#descriptive-statistics)
on the dataframe, such as `.mean()` or `.max()`.

For example, we can calculate the average real minimum wage for each
country over the period 2006 to 2016 (the default is to aggregate over
rows)

```python3 hide-output=false
merged.mean().head(10)
```

Using this series, we can plot the average real minimum wage over the
past decade for each country in our data set

```python3 hide-output=false
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('seaborn')

merged.mean().sort_values(ascending=False).plot(kind='bar', title="Average real minimum wage 2006 - 2016")

#Set country labels
country_labels = merged.mean().sort_values(ascending=False).index.get_level_values('Country').tolist()
plt.xticks(range(0, len(country_labels)), country_labels)
plt.xlabel('Country')

plt.show()
```

Passing in `axis=1` to `.mean()` will aggregate over columns (giving
the average minimum wage for all countries over time)

```python3 hide-output=false
merged.mean(axis=1).head()
```

We can plot this time series as a line graph

```python3 hide-output=false
merged.mean(axis=1).plot()
plt.title('Average real minimum wage 2006 - 2016')
plt.ylabel('2015 USD')
plt.xlabel('Year')
plt.show()
```

We can also specify a level of the `MultiIndex` (in the column axis)
to aggregate over

```python3 hide-output=false
merged.mean(level='Continent', axis=1).head()
```

We can plot the average minimum wages in each continent as a time series

```python3 hide-output=false
merged.mean(level='Continent', axis=1).plot()
plt.title('Average real minimum wage')
plt.ylabel('2015 USD')
plt.xlabel('Year')
plt.show()
```

We will drop Australia as a continent for plotting purposes

```python3 hide-output=false
merged = merged.drop('Australia', level='Continent', axis=1)
merged.mean(level='Continent', axis=1).plot()
plt.title('Average real minimum wage')
plt.ylabel('2015 USD')
plt.xlabel('Year')
plt.show()
```

`.describe()` is useful for quickly retrieving a number of common
summary statistics

```python3 hide-output=false
merged.stack().describe()
```

<!-- #region -->
This is a simplified way to use `groupby`.

Using `groupby` generally follows a ‘split-apply-combine’ process:

- split: data is grouped based on one or more keys  
- apply: a function is called on each group independently  
- combine: the results of the function calls are combined into a new data structure  


The `groupby` method achieves the first step of this process, creating
a new `DataFrameGroupBy` object with data split into groups.

Let’s split `merged` by continent again, this time using the
`groupby` function, and name the resulting object `grouped`
<!-- #endregion -->

```python3 hide-output=false
grouped = merged.groupby(level='Continent', axis=1)
grouped
```

Calling an aggregation method on the object applies the function to each
group, the results of which are combined in a new data structure.

For example, we can return the number of countries in our dataset for
each continent using `.size()`.

In this case, our new data structure is a `Series`

```python3 hide-output=false
grouped.size()
```

Calling `.get_group()` to return just the countries in a single group,
we can create a kernel density estimate of the distribution of real
minimum wages in 2016 for each continent.

`grouped.groups.keys()` will return the keys from the `groupby`
object

```python3 hide-output=false
import seaborn as sns

continents = grouped.groups.keys()

for continent in continents:
    sns.kdeplot(grouped.get_group(continent)['2015'].unstack(), label=continent, shade=True)

plt.title('Real minimum wages in 2015')
plt.xlabel('US dollars')
plt.show()
```

## Final Remarks

This lecture has provided an introduction to some of pandas’ more
advanced features, including multiindices, merging, grouping and
plotting.

Other tools that may be useful in panel data analysis include [xarray](http://xarray.pydata.org/en/stable/), a python package that
extends pandas to N-dimensional data structures.


## Exercises


### Exercise 1

In these exercises, you’ll work with a dataset of employment rates
in Europe by age and sex from [Eurostat](http://ec.europa.eu/eurostat/data/database).

The dataset can be accessed with the following link:

```python3 hide-output=false
url3 = 'https://raw.githubusercontent.com/QuantEcon/lecture-source-py/master/source/_static/lecture_specific/pandas_panel/employ.csv'
```

Reading in the CSV file returns a panel dataset in long format. Use `.pivot_table()` to construct
a wide format dataframe with a `MultiIndex` in the columns.

Start off by exploring the dataframe and the variables available in the
`MultiIndex` levels.

Write a program that quickly returns all values in the `MultiIndex`.


### Exercise 2

Filter the above dataframe to only include employment as a percentage of
‘active population’.

Create a grouped boxplot using `seaborn` of employment rates in 2015
by age group and sex.

**Hint:** `GEO` includes both areas and countries.


## Solutions


### Exercise 1

```python3 hide-output=false
employ = pd.read_csv(url3)
employ = employ.pivot_table(values='Value',
                            index=['DATE'],
                            columns=['UNIT','AGE', 'SEX', 'INDIC_EM', 'GEO'])
employ.index = pd.to_datetime(employ.index) # ensure that dates are datetime format
employ.head()
```

This is a large dataset so it is useful to explore the levels and
variables available

```python3 hide-output=false
employ.columns.names
```

Variables within levels can be quickly retrieved with a loop

```python3 hide-output=false
for name in employ.columns.names:
    print(name, employ.columns.get_level_values(name).unique())
```

### Exercise 2

To easily filter by country, swap `GEO` to the top level and sort the
`MultiIndex`

```python3 hide-output=false
employ.columns = employ.columns.swaplevel(0,-1)
employ = employ.sort_index(axis=1)
```

We need to get rid of a few items in `GEO` which are not countries.

A fast way to get rid of the EU areas is to use a list comprehension to
find the level values in `GEO` that begin with ‘Euro’

```python3 hide-output=false
geo_list = employ.columns.get_level_values('GEO').unique().tolist()
countries = [x for x in geo_list if not x.startswith('Euro')]
employ = employ[countries]
employ.columns.get_level_values('GEO').unique()
```

Select only percentage employed in the active population from the
dataframe

```python3 hide-output=false
employ_f = employ.xs(('Percentage of total population', 'Active population'),
                     level=('UNIT', 'INDIC_EM'),
                     axis=1)
employ_f.head()
```

Drop the ‘Total’ value before creating the grouped boxplot

```python3 hide-output=false
employ_f = employ_f.drop('Total', level='SEX', axis=1)
```

```python3 hide-output=false
box = employ_f['2015'].unstack().reset_index()
sns.boxplot(x="AGE", y=0, hue="SEX", data=box, palette=("husl"), showfliers=False)
plt.xlabel('')
plt.xticks(rotation=35)
plt.ylabel('Percentage of population (%)')
plt.title('Employment in Europe (2015)')
plt.legend(bbox_to_anchor=(1,0.5))
plt.show()
```
