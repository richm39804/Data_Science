
# theDevMasters, Day 2:<br> Data Preparation and Exploration with Visualization + External Data Acquisition

Company Name : theDevMasters <br>
Author: Cloris Li <br>
Reviewed: Zia Khan <br>
MDS-DL Document Version : 1.0 <br>

Description : Day 2, Data Preparation and Exploration with Visualization + External Data Acquisition

# <font color="blue">Table of Contents</font>

## Data Preparation and Exploration
* Feature Engineering
* Missing Value Imputation
* Creating Dummies

## Data Visualization
* Libraries:
    * Matplotlib
    * Pandas Visualization
    * Seaborn
    * Tableau
* Common used plots:
     * Scatter plot
     * Line chart
     * Histogram
     * Bar chart
     * Box plot
     * Heatmap
     * Pair plot
     * Faceting

## Data Acquisition (External)
* Web Scraping
* APIs

## Hack Project: Election Day
* Project description

# <font color="red">I. Data Preparation and Exploration </font>

### 1.0 Data Collection and Understanding


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings('ingore')
```


```python
%matplotlib inline
```

#### Data Dictionary:
* PassengerId - A numerical id assigned to each passenger
* Survived - Whether the passenger survived (1), or didn't (0). This is going to be the dependent variable of our study
* Pclass - The class the passenger was in - first class (1), second class (2), or third class (3). Pclass is going to be one of the independent variables in our study
* Name - the name of the passenger
* Sex - The gender of the passenger - male or female. Sex is going to be one of the independent variables in our study
* Age - The age of the passenger. Fractional. Age (of age groups) is going to be one of the independent variables in our study
* SibSp - The number of siblings and spouses the passenger had on board
* Parch - The number of parents and children the passenger had on board
* Ticket - The ticket number of the passenger
* Fare - How much the passenger paid for the ticker
* Cabin - Which cabin the passenger was in
* Embarked - Where the passenger boarded the Titanic


```python
x = np.arange(20)
y = np.random.normal(10, 1, 20)
z = np.random.normal(10, 2, 20)
# line above esc a line below esc b
```


```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.set_title('Line chart')
ax.set_xlabel('x')
ax.set_ylabel('y')
```




    Text(0, 0.5, 'y')




![png](output_9_1.png)


### 1.1 Feature Engineering
Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. Feature engineering can substantially boost machine learning model performance.

#### 1.1.1 Name to Title


```python

```


```python
# another option:

```


```python
# mapping

```

#### 1.1.2 Family size


```python

```

### <font color="green">Q: Can you try feature engineering on another feature?</font>


```python

```

### 1.2 Missing Value Imputation
Different types of missing values:
* Not Missing at Random: NMAR
* Missing at Random: MAR
* Missing Completely at Random: MCAR

There are various ways we can imputate the missing values. We can replace the missing/null values with either of 3 M’s (**Mean/ Mode/ Median**) depending on the possible values of the given column, or simply drop the missing values. We can also fill in missing values using prediction model (such as regression).

#### 1.2.1 missing Age


```python

```


```python

```


```python

```

### <font color="green">Q: How do you decide which of 3'M to use for missing value imputation?</font>


```python

```

#### 1.2.2 missing Embarked


```python

```


```python
# fill out missing embark with S embark

```


```python

```

### 1.3 Creating Dummies
A dummy variable is a placeholder for a variable that will be integrated over, summed over, or marginalized.   However, in machine learning, it often describes the individual variables in a one-hot encoding scheme. Thus, dummy or Boolean variables are qualitative variables that can only take the value 0 or 1 to indicate the absence or presence of a specified condition.


```python

```


```python

```

### <font color="green">Q: Why should we set "drop_first = True"?</font>


```python

```

# <font color="red">II. Data Visualization </font>

Data visualization is the discipline of trying to understand data by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be exposed.

Python offers multiple great graphing libraries that come packed with lots of different features.

Here we will introduce a few popular plotting libraries:
* Matplotlib: low level, provides lots of freedom
* Pandas Visualization: easy to use interface, built on Matplotlib
* Seaborn: high-level interface, great default styles

### <font color="green">Q: In your own words, why is visualization important?</font>


```python

```

## 2.0 Import data and graphing libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn
seaborn.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
```


```python

```


```python

```

### <font color="green">Q: What happens if you do not run %matplotlib inline?</font>


```python
%matplotlib inline
```

## 2.1 Matplotlib
Matplotlib is the most popular python plotting library. It is a low-level library with a Matlab like interface which offers lots of freedom at the cost of having to write more code.


```python
x = np.arange(20)
y = np.random.normal(10, 1, 20)
z = np.random.normal(10, 2, 20)
```


```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.set_title('Line chart')
ax.set_xlabel('x')
ax.set_ylabel('y')
```




    Text(0, 0.5, 'y')




![png](output_47_1.png)



```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, label = 'Prediction')
ax.plot(x, z, label = 'Actual Value')
ax.legend(loc=0)
plt.grid()

```


![png](output_48_0.png)



```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, marker = 'o', c = '#FFF000')
ax.set_ylabel('$$$ Money', fontsize=10)
ax.set_xlabel('Years', fontsize=20)
fig.savefig('figure.png', dpi=300)




```


![png](output_49_0.png)



```python
palm = plt.figure(figsize = (10, 6))
a = palm.add_subplot(121)
a.plot(x, y)
b = palm.add_subplot(122)
b.scatter(x,y);
```


![png](output_50_0.png)



```python

```

### <font color="green">Q: Why is it important for all of your data to be the same length?</font>


```python

```


```python

```


```python

```


```python

```


```python

```


```python
fig2 = plt.figure(figsize = (15,5))
a = fig2.add_subplot(131)
a.scatter(x,y)
a.set_title('Scatter plot')

b = fig2.add_subplot(132)
b.hist(y)
b.set_title('Histogram')

c = fig2.add_subplot(133)
c.boxplot([y,z])
c.set_title('Box plot')

fig2.suptitle('Visualization')
```




    Text(0.5, 0.98, 'Visualization')




![png](output_58_1.png)



```python
oak = plt.figure()
first = oak.add_subplot(211)
second = oak.add_subplot(212)
first.hist(y)
second.boxplot([y,z]);
```


![png](output_59_0.png)


### <font color="green">Q: What is the main difference between a line plot & a scatter plot?</font>


```python

```

## <font color="green">Exercises</font>
1. Create a figure with 4 subplots.
    1. Histogram of Ages of Titanic
    2. Scatterplot of Age vs Fares
    3. Boxplot of Ages of Survivors vs Deceased
    4. Boxplot of Ages of Men vs Women
2. Interpret each chart you created.
    1. Does it show you anything interesting?
    2. Does it help you figure out who will survive
3. Pick a chart from the matplotlib gallery & recreate it in your notebook.


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np



data=pd.read_csv('train.csv')
notMissing=data[data['Age'].notnull()]
ship = plt.figure(figsize = (12,10))
#
hist = ship.add_subplot(221)
hist.hist(notMissing['Age'])
#
scatter = ship.add_subplot(222)
scatter.scatter(notMissing['Age'],notMissing['Fare'])
#
box1 = ship.add_subplot(223)
box1.boxplot([notMissing[notMissing['Survived']==0]['Age'],
             notMissing[notMissing['Survived']==1]['Age']],
           labels = ['Deceased','Survived']);
#
box2 = ship.add_subplot(224)
box2.boxplot([notMissing[notMissing['Sex']=='male']['Age'],
             notMissing[notMissing['Sex']=='female']['Age']],
           labels = ['Male','Female']);

```


![png](output_63_0.png)



```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.plot(x, np.cos(z))

plt.show()
```


![png](output_65_0.png)



```python
fig = plt.figure()
ax = plt.axes()
z = np.linspace(10, 20, 100)
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x), c='red');
ax.plot(x, np.sin(x), c='black');
ax.plot(X, np.sin(z), c='blue');
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-58-439dc98b5663> in <module>
          5 ax.plot(x, np.sin(x), c='red');
          6 ax.plot(x, np.sin(x), c='black');
    ----> 7 ax.plot(X, np.sin(z), c='blue');
    

    NameError: name 'X' is not defined



![png](output_66_1.png)



```python



plt.plot(x, np.sin(x = 0), color='blue')
plt.plot(x, np.sin(x = 1), color='g')
plt.plot(x, np.sin(x = 2), color='0.75')
plt.plot(x, np.sin(x = 3), color='#FFDD44')
plt.plot(x, np.sin(x = 4), color=('1.0,0.2,0.3'))
plt.plot(x, np.sin(x = 5), color='chartreuse', linestyle=':');
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-62-9923db7d7110> in <module>
    ----> 1 plt.plot(x, np.sin(x = 0), color='blue')
          2 plt.plot(x, np.sin(x = 1), color='g')
          3 plt.plot(x, np.sin(x = 2), color='0.75')
          4 plt.plot(x, np.sin(x = 3), color='#FFDD44')
          5 plt.plot(x, np.sin(x = 4), color=('1.0,0.2,0.3'))


    ValueError: invalid number of arguments



```python

```


```python

```

## 2.2 Pandas Visualization
Pandas is an open source high-performance, easy-to-use library providing data structures, such as dataframes, and data analysis tools like the visualization tools. Pandas Visualization makes it really easy to create plots out of a pandas dataframe and series. It also has a higher level API than Matplotlib and therefore we need less code for the same results.


```python

```


```python

```


```python

```

### <font color="green">Q: What do we use a histogram for?</font>


```python

```


```python

```


```python

```


```python

```

### <font color="green">Q: What do we use a boxplot for?</font>


```python

```

## 2.3 Seaborn
Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for creating attractive graphs. Seaborn has a lot to offer. You can create graphs in one line that would take you multiple tens of lines in Matplotlib. Its standard designs are awesome and it also has a nice interface for working with pandas dataframes.


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
```


```python
data=pd.read_csv('train.csv')
notMissing=data[data['Age'].notnull()]
ship = plt.figure(figsize = (12,10))
#
hist = ship.add_subplot(221)
hist.hist(notMissing['Age'])
#
scatter = ship.add_subplot(222)
scatter.scatter(notMissing['Age'],notMissing['Fare'])
#
box1 = ship.add_subplot(223)
box1.boxplot([notMissing[notMissing['Survived']==0]['Age'],
             notMissing[notMissing['Survived']==1]['Age']],
           labels = ['Deceased','Survived']);
#
box2 = ship.add_subplot(224)
box2.boxplot([notMissing[notMissing['Sex']=='male']['Age'],
             notMissing[notMissing['Sex']=='female']['Age']],
           labels = ['Male','Female']);


```


![png](output_83_0.png)



```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>-0.005007</td>
      <td>-0.035144</td>
      <td>0.036847</td>
      <td>-0.057527</td>
      <td>-0.001652</td>
      <td>0.012658</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.005007</td>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.035144</td>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.036847</td>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.057527</td>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.001652</td>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.012658</td>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
import seaborn
seaborn.set_style('whitegrid')

```


```python
seaborn.heatmap(data.corr(), annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2020ccf8>




![png](output_87_1.png)


## Data Science - everything is a vector


```python
seaborn.pairplot(notMissing, hue ='Survived')
```




    <seaborn.axisgrid.PairGrid at 0x1a23e047b8>




![png](output_89_1.png)



```python
import seaborn as sns
sns.set
df=sns.load_dataset('iris')
sns.pairplot(df,hue='species')
```




    <seaborn.axisgrid.PairGrid at 0x1a26294e10>




![png](output_90_1.png)



```python
import matplotlib.pyplot as plt
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0, 0.1, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
```


![png](output_91_0.png)



```python
#swarm plot
iris=sns.load_dataset('iris')
sns.swarmplot(x='species', y='petal_length', data=iris)
plt.show()
```


![png](output_92_0.png)



```python
data.plot.scatter(x='Age', y='Fare', title='Scatter plot')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a279f5390>




![png](output_93_1.png)



```python
data[['Age', 'Fare']].plot.line(title='Line chart')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a279f5fd0>




![png](output_94_1.png)



```python
data('Age').plot.hist
```


```python
data['Pclass'].value_counts().sort_index().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27d64828>




![png](output_96_1.png)



```python
data['Pclass'].value_counts().sort_index().plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27f74710>




![png](output_97_1.png)



```python
data['Age'].plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27f67a58>




![png](output_98_1.png)



```python
import seaborn as sns

```


```python
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a28197a90>




![png](output_100_1.png)



```python
sns.lineplot(data=data[['Age','Fare']])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a283ece10>




![png](output_101_1.png)



```python
sns.distplot(data['Age'], bins=10, kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a279c19b0>




![png](output_102_1.png)



```python
sns.countplot(data['Embarked'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27430c18>




![png](output_103_1.png)



```python
sns.boxplot('Age', 'Sex', data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27a76b00>




![png](output_104_1.png)



```python
g = sns.FacetGrid(data, col='Survived')
g = g.map(sns.kdeplot, 'Pclass')
```


![png](output_105_0.png)



```python
new=pd.get_dummies(data)
```

### <font color="green">Q: Is there anything interesting in the heatmap?</font>


```python

```


```python

```


```python

```

### <font color="green">Q: Is there anything to note from the pair plot?</font>


```python

```

# <font color="red">III. Data Acqisition (External) </font>

A data analyst or data scientist doesn’t always get data handed to them in a CSV or via an easily accessible database. Sometimes, you’ve got to go out and get the data you need. The ability to collect unique data sets can really set you apart from the pack, and being able to access APIs and scrape the web for new data stories is the best way to get data nobody else is working with.

## 3.1 Web Scraping
Web scraping consists in gathering data available on websites. This can be done manually by a human user or by a bot.

Web scrapers gather website data in the same way a human would do it: the scraper goes onto a web page of the website, gets the relevant data, and move forward to the next web page. Every website has a different structure, that is why web scrapers are usually built to explore one website.

Websites are created using HTML (Hypertext Markup Language), along with CSS (Cascading Style Sheets) and JavaScript. HTML elements are separated by tags and they directly introduce content to the web page.

BeautifulSoup will be used here to parse the HTML files. It is one of the most used library for web scraping. Its is quite simple to use and has many features that help gathering websites data efficiently. (Documentation see: https://www.crummy.com/software/BeautifulSoup/bs4/doc/)


### <font color="green">Q: When can you not webscrap a page?</font>


```python

```


```python
url = 'https://www.coinmarketcap.com'
```


```python
!pip install requests
```

    Requirement already satisfied: requests in ./anaconda3/lib/python3.7/site-packages (2.22.0)
    Requirement already satisfied: certifi>=2017.4.17 in ./anaconda3/lib/python3.7/site-packages (from requests) (2019.6.16)
    Requirement already satisfied: idna<2.9,>=2.5 in ./anaconda3/lib/python3.7/site-packages (from requests) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in ./anaconda3/lib/python3.7/site-packages (from requests) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in ./anaconda3/lib/python3.7/site-packages (from requests) (1.24.2)



```python
import requests
```


```python
requests.get(url)
```




    <Response [200]>




```python
m = requests.get(url)
m.content
```




    b'<!doctype html>\n<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7"> <![endif]-->\n<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8"> <![endif]-->\n<!--[if IE 8]>         <html class="no-js lt-ie9"> <![endif]-->\n<!--[if gt IE 8]><!-->\n<html lang="en"> <!--<![endif]-->\n<head>\n<meta charset="utf-8">\n<meta http-equiv="x-ua-compatible" content="ie=edge"><script type="text/javascript">(window.NREUM||(NREUM={})).loader_config={xpid:"VQ4BV1dWDxACVVZTAAUEUVI="};window.NREUM||(NREUM={}),__nr_require=function(t,e,n){function r(n){if(!e[n]){var o=e[n]={exports:{}};t[n][0].call(o.exports,function(e){var o=t[n][1][e];return r(o||e)},o,o.exports)}return e[n].exports}if("function"==typeof __nr_require)return __nr_require;for(var o=0;o<n.length;o++)r(n[o]);return r}({1:[function(t,e,n){function r(t){try{s.console&&console.log(t)}catch(e){}}var o,i=t("ee"),a=t(23),s={};try{o=localStorage.getItem("__nr_flags").split(","),console&&"function"==typeof console.log&&(s.console=!0,o.indexOf("dev")!==-1&&(s.dev=!0),o.indexOf("nr_dev")!==-1&&(s.nrDev=!0))}catch(c){}s.nrDev&&i.on("internal-error",function(t){r(t.stack)}),s.dev&&i.on("fn-err",function(t,e,n){r(n.stack)}),s.dev&&(r("NR AGENT IN DEVELOPMENT MODE"),r("flags: "+a(s,function(t,e){return t}).join(", ")))},{}],2:[function(t,e,n){function r(t,e,n,r,s){try{l?l-=1:o(s||new UncaughtException(t,e,n),!0)}catch(f){try{i("ierr",[f,c.now(),!0])}catch(d){}}return"function"==typeof u&&u.apply(this,a(arguments))}function UncaughtException(t,e,n){this.message=t||"Uncaught error with no additional information",this.sourceURL=e,this.line=n}function o(t,e){var n=e?null:c.now();i("err",[t,n])}var i=t("handle"),a=t(24),s=t("ee"),c=t("loader"),f=t("gos"),u=window.onerror,d=!1,p="nr@seenError",l=0;c.features.err=!0,t(1),window.onerror=r;try{throw new Error}catch(h){"stack"in h&&(t(13),t(12),"addEventListener"in window&&t(6),c.xhrWrappable&&t(14),d=!0)}s.on("fn-start",function(t,e,n){d&&(l+=1)}),s.on("fn-err",function(t,e,n){d&&!n[p]&&(f(n,p,function(){return!0}),this.thrown=!0,o(n))}),s.on("fn-end",function(){d&&!this.thrown&&l>0&&(l-=1)}),s.on("internal-error",function(t){i("ierr",[t,c.now(),!0])})},{}],3:[function(t,e,n){t("loader").features.ins=!0},{}],4:[function(t,e,n){function r(){j++,L=y.hash,this[u]=x.now()}function o(){j--,y.hash!==L&&i(0,!0);var t=x.now();this[h]=~~this[h]+t-this[u],this[d]=t}function i(t,e){E.emit("newURL",[""+y,e])}function a(t,e){t.on(e,function(){this[e]=x.now()})}var s="-start",c="-end",f="-body",u="fn"+s,d="fn"+c,p="cb"+s,l="cb"+c,h="jsTime",m="fetch",v="addEventListener",w=window,y=w.location,x=t("loader");if(w[v]&&x.xhrWrappable){var g=t(10),b=t(11),E=t(8),R=t(6),O=t(13),C=t(7),P=t(14),T=t(9),N=t("ee"),S=N.get("tracer");t(16),x.features.spa=!0;var L,j=0;N.on(u,r),N.on(p,r),N.on(d,o),N.on(l,o),N.buffer([u,d,"xhr-done","xhr-resolved"]),R.buffer([u]),O.buffer(["setTimeout"+c,"clearTimeout"+s,u]),P.buffer([u,"new-xhr","send-xhr"+s]),C.buffer([m+s,m+"-done",m+f+s,m+f+c]),E.buffer(["newURL"]),g.buffer([u]),b.buffer(["propagate",p,l,"executor-err","resolve"+s]),S.buffer([u,"no-"+u]),T.buffer(["new-jsonp","cb-start","jsonp-error","jsonp-end"]),a(P,"send-xhr"+s),a(N,"xhr-resolved"),a(N,"xhr-done"),a(C,m+s),a(C,m+"-done"),a(T,"new-jsonp"),a(T,"jsonp-end"),a(T,"cb-start"),E.on("pushState-end",i),E.on("replaceState-end",i),w[v]("hashchange",i,!0),w[v]("load",i,!0),w[v]("popstate",function(){i(0,j>1)},!0)}},{}],5:[function(t,e,n){function r(t){}if(window.performance&&window.performance.timing&&window.performance.getEntriesByType){var o=t("ee"),i=t("handle"),a=t(13),s=t(12),c="learResourceTimings",f="addEventListener",u="resourcetimingbufferfull",d="bstResource",p="resource",l="-start",h="-end",m="fn"+l,v="fn"+h,w="bstTimer",y="pushState",x=t("loader");x.features.stn=!0,t(8);var g=NREUM.o.EV;o.on(m,function(t,e){var n=t[0];n instanceof g&&(this.bstStart=x.now())}),o.on(v,function(t,e){var n=t[0];n instanceof g&&i("bst",[n,e,this.bstStart,x.now()])}),a.on(m,function(t,e,n){this.bstStart=x.now(),this.bstType=n}),a.on(v,function(t,e){i(w,[e,this.bstStart,x.now(),this.bstType])}),s.on(m,function(){this.bstStart=x.now()}),s.on(v,function(t,e){i(w,[e,this.bstStart,x.now(),"requestAnimationFrame"])}),o.on(y+l,function(t){this.time=x.now(),this.startPath=location.pathname+location.hash}),o.on(y+h,function(t){i("bstHist",[location.pathname+location.hash,this.startPath,this.time])}),f in window.performance&&(window.performance["c"+c]?window.performance[f](u,function(t){i(d,[window.performance.getEntriesByType(p)]),window.performance["c"+c]()},!1):window.performance[f]("webkit"+u,function(t){i(d,[window.performance.getEntriesByType(p)]),window.performance["webkitC"+c]()},!1)),document[f]("scroll",r,{passive:!0}),document[f]("keypress",r,!1),document[f]("click",r,!1)}},{}],6:[function(t,e,n){function r(t){for(var e=t;e&&!e.hasOwnProperty(u);)e=Object.getPrototypeOf(e);e&&o(e)}function o(t){s.inPlace(t,[u,d],"-",i)}function i(t,e){return t[1]}var a=t("ee").get("events"),s=t(26)(a,!0),c=t("gos"),f=XMLHttpRequest,u="addEventListener",d="removeEventListener";e.exports=a,"getPrototypeOf"in Object?(r(document),r(window),r(f.prototype)):f.prototype.hasOwnProperty(u)&&(o(window),o(f.prototype)),a.on(u+"-start",function(t,e){var n=t[1],r=c(n,"nr@wrapped",function(){function t(){if("function"==typeof n.handleEvent)return n.handleEvent.apply(n,arguments)}var e={object:t,"function":n}[typeof n];return e?s(e,"fn-",null,e.name||"anonymous"):n});this.wrapped=t[1]=r}),a.on(d+"-start",function(t){t[1]=this.wrapped||t[1]})},{}],7:[function(t,e,n){function r(t,e,n){var r=t[e];"function"==typeof r&&(t[e]=function(){var t=r.apply(this,arguments);return o.emit(n+"start",arguments,t),t.then(function(e){return o.emit(n+"end",[null,e],t),e},function(e){throw o.emit(n+"end",[e],t),e})})}var o=t("ee").get("fetch"),i=t(23);e.exports=o;var a=window,s="fetch-",c=s+"body-",f=["arrayBuffer","blob","json","text","formData"],u=a.Request,d=a.Response,p=a.fetch,l="prototype";u&&d&&p&&(i(f,function(t,e){r(u[l],e,c),r(d[l],e,c)}),r(a,"fetch",s),o.on(s+"end",function(t,e){var n=this;if(e){var r=e.headers.get("content-length");null!==r&&(n.rxSize=r),o.emit(s+"done",[null,e],n)}else o.emit(s+"done",[t],n)}))},{}],8:[function(t,e,n){var r=t("ee").get("history"),o=t(26)(r);e.exports=r;var i=window.history&&window.history.constructor&&window.history.constructor.prototype,a=window.history;i&&i.pushState&&i.replaceState&&(a=i),o.inPlace(a,["pushState","replaceState"],"-")},{}],9:[function(t,e,n){function r(t){function e(){c.emit("jsonp-end",[],p),t.removeEventListener("load",e,!1),t.removeEventListener("error",n,!1)}function n(){c.emit("jsonp-error",[],p),c.emit("jsonp-end",[],p),t.removeEventListener("load",e,!1),t.removeEventListener("error",n,!1)}var r=t&&"string"==typeof t.nodeName&&"script"===t.nodeName.toLowerCase();if(r){var o="function"==typeof t.addEventListener;if(o){var a=i(t.src);if(a){var u=s(a),d="function"==typeof u.parent[u.key];if(d){var p={};f.inPlace(u.parent,[u.key],"cb-",p),t.addEventListener("load",e,!1),t.addEventListener("error",n,!1),c.emit("new-jsonp",[t.src],p)}}}}}function o(){return"addEventListener"in window}function i(t){var e=t.match(u);return e?e[1]:null}function a(t,e){var n=t.match(p),r=n[1],o=n[3];return o?a(o,e[r]):e[r]}function s(t){var e=t.match(d);return e&&e.length>=3?{key:e[2],parent:a(e[1],window)}:{key:t,parent:window}}var c=t("ee").get("jsonp"),f=t(26)(c);if(e.exports=c,o()){var u=/[?&](?:callback|cb)=([^&#]+)/,d=/(.*)\\.([^.]+)/,p=/^(\\w+)(\\.|$)(.*)$/,l=["appendChild","insertBefore","replaceChild"];Node&&Node.prototype&&Node.prototype.appendChild?f.inPlace(Node.prototype,l,"dom-"):(f.inPlace(HTMLElement.prototype,l,"dom-"),f.inPlace(HTMLHeadElement.prototype,l,"dom-"),f.inPlace(HTMLBodyElement.prototype,l,"dom-")),c.on("dom-start",function(t){r(t[0])})}},{}],10:[function(t,e,n){var r=t("ee").get("mutation"),o=t(26)(r),i=NREUM.o.MO;e.exports=r,i&&(window.MutationObserver=function(t){return this instanceof i?new i(o(t,"fn-")):i.apply(this,arguments)},MutationObserver.prototype=i.prototype)},{}],11:[function(t,e,n){function r(t){var e=a.context(),n=s(t,"executor-",e),r=new f(n);return a.context(r).getCtx=function(){return e},a.emit("new-promise",[r,e],e),r}function o(t,e){return e}var i=t(26),a=t("ee").get("promise"),s=i(a),c=t(23),f=NREUM.o.PR;e.exports=a,f&&(window.Promise=r,["all","race"].forEach(function(t){var e=f[t];f[t]=function(n){function r(t){return function(){a.emit("propagate",[null,!o],i),o=o||!t}}var o=!1;c(n,function(e,n){Promise.resolve(n).then(r("all"===t),r(!1))});var i=e.apply(f,arguments),s=f.resolve(i);return s}}),["resolve","reject"].forEach(function(t){var e=f[t];f[t]=function(t){var n=e.apply(f,arguments);return t!==n&&a.emit("propagate",[t,!0],n),n}}),f.prototype["catch"]=function(t){return this.then(null,t)},f.prototype=Object.create(f.prototype,{constructor:{value:r}}),c(Object.getOwnPropertyNames(f),function(t,e){try{r[e]=f[e]}catch(n){}}),a.on("executor-start",function(t){t[0]=s(t[0],"resolve-",this),t[1]=s(t[1],"resolve-",this)}),a.on("executor-err",function(t,e,n){t[1](n)}),s.inPlace(f.prototype,["then"],"then-",o),a.on("then-start",function(t,e){this.promise=e,t[0]=s(t[0],"cb-",this),t[1]=s(t[1],"cb-",this)}),a.on("then-end",function(t,e,n){this.nextPromise=n;var r=this.promise;a.emit("propagate",[r,!0],n)}),a.on("cb-end",function(t,e,n){a.emit("propagate",[n,!0],this.nextPromise)}),a.on("propagate",function(t,e,n){this.getCtx&&!e||(this.getCtx=function(){if(t instanceof Promise)var e=a.context(t);return e&&e.getCtx?e.getCtx():this})}),r.toString=function(){return""+f})},{}],12:[function(t,e,n){var r=t("ee").get("raf"),o=t(26)(r),i="equestAnimationFrame";e.exports=r,o.inPlace(window,["r"+i,"mozR"+i,"webkitR"+i,"msR"+i],"raf-"),r.on("raf-start",function(t){t[0]=o(t[0],"fn-")})},{}],13:[function(t,e,n){function r(t,e,n){t[0]=a(t[0],"fn-",null,n)}function o(t,e,n){this.method=n,this.timerDuration=isNaN(t[1])?0:+t[1],t[0]=a(t[0],"fn-",this,n)}var i=t("ee").get("timer"),a=t(26)(i),s="setTimeout",c="setInterval",f="clearTimeout",u="-start",d="-";e.exports=i,a.inPlace(window,[s,"setImmediate"],s+d),a.inPlace(window,[c],c+d),a.inPlace(window,[f,"clearImmediate"],f+d),i.on(c+u,r),i.on(s+u,o)},{}],14:[function(t,e,n){function r(t,e){d.inPlace(e,["onreadystatechange"],"fn-",s)}function o(){var t=this,e=u.context(t);t.readyState>3&&!e.resolved&&(e.resolved=!0,u.emit("xhr-resolved",[],t)),d.inPlace(t,y,"fn-",s)}function i(t){x.push(t),h&&(b?b.then(a):v?v(a):(E=-E,R.data=E))}function a(){for(var t=0;t<x.length;t++)r([],x[t]);x.length&&(x=[])}function s(t,e){return e}function c(t,e){for(var n in t)e[n]=t[n];return e}t(6);var f=t("ee"),u=f.get("xhr"),d=t(26)(u),p=NREUM.o,l=p.XHR,h=p.MO,m=p.PR,v=p.SI,w="readystatechange",y=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"],x=[];e.exports=u;var g=window.XMLHttpRequest=function(t){var e=new l(t);try{u.emit("new-xhr",[e],e),e.addEventListener(w,o,!1)}catch(n){try{u.emit("internal-error",[n])}catch(r){}}return e};if(c(l,g),g.prototype=l.prototype,d.inPlace(g.prototype,["open","send"],"-xhr-",s),u.on("send-xhr-start",function(t,e){r(t,e),i(e)}),u.on("open-xhr-start",r),h){var b=m&&m.resolve();if(!v&&!m){var E=1,R=document.createTextNode(E);new h(a).observe(R,{characterData:!0})}}else f.on("fn-end",function(t){t[0]&&t[0].type===w||a()})},{}],15:[function(t,e,n){function r(){var t=window.NREUM,e=t.info.accountID||null,n=t.info.agentID||null,r=t.info.trustKey||null,i="btoa"in window&&"function"==typeof window.btoa;if(!e||!n||!i)return null;var a={v:[0,1],d:{ty:"Browser",ac:e,ap:n,id:o.generateCatId(),tr:o.generateCatId(),ti:Date.now()}};return r&&e!==r&&(a.d.tk=r),btoa(JSON.stringify(a))}var o=t(21);e.exports={generateTraceHeader:r}},{}],16:[function(t,e,n){function r(t){var e=this.params,n=this.metrics;if(!this.ended){this.ended=!0;for(var r=0;r<l;r++)t.removeEventListener(p[r],this.listener,!1);e.aborted||(n.duration=s.now()-this.startTime,this.loadCaptureCalled||4!==t.readyState?null==e.status&&(e.status=0):a(this,t),n.cbTime=this.cbTime,d.emit("xhr-done",[t],t),c("xhr",[e,n,this.startTime]))}}function o(t,e){var n=t.responseType;if("json"===n&&null!==e)return e;var r="arraybuffer"===n||"blob"===n||"json"===n?t.response:t.responseText;return v(r)}function i(t,e){var n=f(e),r=t.params;r.host=n.hostname+":"+n.port,r.pathname=n.pathname,t.sameOrigin=n.sameOrigin}function a(t,e){t.params.status=e.status;var n=o(e,t.lastSize);if(n&&(t.metrics.rxSize=n),t.sameOrigin){var r=e.getResponseHeader("X-NewRelic-App-Data");r&&(t.params.cat=r.split(", ").pop())}t.loadCaptureCalled=!0}var s=t("loader");if(s.xhrWrappable){var c=t("handle"),f=t(17),u=t(15).generateTraceHeader,d=t("ee"),p=["load","error","abort","timeout"],l=p.length,h=t("id"),m=t(20),v=t(19),w=window.XMLHttpRequest;s.features.xhr=!0,t(14),d.on("new-xhr",function(t){var e=this;e.totalCbs=0,e.called=0,e.cbTime=0,e.end=r,e.ended=!1,e.xhrGuids={},e.lastSize=null,e.loadCaptureCalled=!1,t.addEventListener("load",function(n){a(e,t)},!1),m&&(m>34||m<10)||window.opera||t.addEventListener("progress",function(t){e.lastSize=t.loaded},!1)}),d.on("open-xhr-start",function(t){this.params={method:t[0]},i(this,t[1]),this.metrics={}}),d.on("open-xhr-end",function(t,e){"loader_config"in NREUM&&"xpid"in NREUM.loader_config&&this.sameOrigin&&e.setRequestHeader("X-NewRelic-ID",NREUM.loader_config.xpid);var n=!1;if("init"in NREUM&&"distributed_tracing"in NREUM.init&&(n=!!NREUM.init.distributed_tracing.enabled),n&&this.sameOrigin){var r=u();r&&e.setRequestHeader("newrelic",r)}}),d.on("send-xhr-start",function(t,e){var n=this.metrics,r=t[0],o=this;if(n&&r){var i=v(r);i&&(n.txSize=i)}this.startTime=s.now(),this.listener=function(t){try{"abort"!==t.type||o.loadCaptureCalled||(o.params.aborted=!0),("load"!==t.type||o.called===o.totalCbs&&(o.onloadCalled||"function"!=typeof e.onload))&&o.end(e)}catch(n){try{d.emit("internal-error",[n])}catch(r){}}};for(var a=0;a<l;a++)e.addEventListener(p[a],this.listener,!1)}),d.on("xhr-cb-time",function(t,e,n){this.cbTime+=t,e?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof n.onload||this.end(n)}),d.on("xhr-load-added",function(t,e){var n=""+h(t)+!!e;this.xhrGuids&&!this.xhrGuids[n]&&(this.xhrGuids[n]=!0,this.totalCbs+=1)}),d.on("xhr-load-removed",function(t,e){var n=""+h(t)+!!e;this.xhrGuids&&this.xhrGuids[n]&&(delete this.xhrGuids[n],this.totalCbs-=1)}),d.on("addEventListener-end",function(t,e){e instanceof w&&"load"===t[0]&&d.emit("xhr-load-added",[t[1],t[2]],e)}),d.on("removeEventListener-end",function(t,e){e instanceof w&&"load"===t[0]&&d.emit("xhr-load-removed",[t[1],t[2]],e)}),d.on("fn-start",function(t,e,n){e instanceof w&&("onload"===n&&(this.onload=!0),("load"===(t[0]&&t[0].type)||this.onload)&&(this.xhrCbStart=s.now()))}),d.on("fn-end",function(t,e){this.xhrCbStart&&d.emit("xhr-cb-time",[s.now()-this.xhrCbStart,this.onload,e],e)})}},{}],17:[function(t,e,n){e.exports=function(t){var e=document.createElement("a"),n=window.location,r={};e.href=t,r.port=e.port;var o=e.href.split("://");!r.port&&o[1]&&(r.port=o[1].split("/")[0].split("@").pop().split(":")[1]),r.port&&"0"!==r.port||(r.port="https"===o[0]?"443":"80"),r.hostname=e.hostname||n.hostname,r.pathname=e.pathname,r.protocol=o[0],"/"!==r.pathname.charAt(0)&&(r.pathname="/"+r.pathname);var i=!e.protocol||":"===e.protocol||e.protocol===n.protocol,a=e.hostname===document.domain&&e.port===n.port;return r.sameOrigin=i&&(!e.hostname||a),r}},{}],18:[function(t,e,n){function r(){}function o(t,e,n){return function(){return i(t,[f.now()].concat(s(arguments)),e?null:this,n),e?void 0:this}}var i=t("handle"),a=t(23),s=t(24),c=t("ee").get("tracer"),f=t("loader"),u=NREUM;"undefined"==typeof window.newrelic&&(newrelic=u);var d=["setPageViewName","setCustomAttribute","setErrorHandler","finished","addToTrace","inlineHit","addRelease"],p="api-",l=p+"ixn-";a(d,function(t,e){u[e]=o(p+e,!0,"api")}),u.addPageAction=o(p+"addPageAction",!0),u.setCurrentRouteName=o(p+"routeName",!0),e.exports=newrelic,u.interaction=function(){return(new r).get()};var h=r.prototype={createTracer:function(t,e){var n={},r=this,o="function"==typeof e;return i(l+"tracer",[f.now(),t,n],r),function(){if(c.emit((o?"":"no-")+"fn-start",[f.now(),r,o],n),o)try{return e.apply(this,arguments)}catch(t){throw c.emit("fn-err",[arguments,this,t],n),t}finally{c.emit("fn-end",[f.now()],n)}}}};a("actionText,setName,setAttribute,save,ignore,onEnd,getContext,end,get".split(","),function(t,e){h[e]=o(l+e)}),newrelic.noticeError=function(t,e){"string"==typeof t&&(t=new Error(t)),i("err",[t,f.now(),!1,e])}},{}],19:[function(t,e,n){e.exports=function(t){if("string"==typeof t&&t.length)return t.length;if("object"==typeof t){if("undefined"!=typeof ArrayBuffer&&t instanceof ArrayBuffer&&t.byteLength)return t.byteLength;if("undefined"!=typeof Blob&&t instanceof Blob&&t.size)return t.size;if(!("undefined"!=typeof FormData&&t instanceof FormData))try{return JSON.stringify(t).length}catch(e){return}}}},{}],20:[function(t,e,n){var r=0,o=navigator.userAgent.match(/Firefox[\\/\\s](\\d+\\.\\d+)/);o&&(r=+o[1]),e.exports=r},{}],21:[function(t,e,n){function r(){function t(){return e?15&e[n++]:16*Math.random()|0}var e=null,n=0,r=window.crypto||window.msCrypto;r&&r.getRandomValues&&(e=r.getRandomValues(new Uint8Array(31)));for(var o,i="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx",a="",s=0;s<i.length;s++)o=i[s],"x"===o?a+=t().toString(16):"y"===o?(o=3&t()|8,a+=o.toString(16)):a+=o;return a}function o(){function t(){return e?15&e[n++]:16*Math.random()|0}var e=null,n=0,r=window.crypto||window.msCrypto;r&&r.getRandomValues&&Uint8Array&&(e=r.getRandomValues(new Uint8Array(31)));for(var o=[],i=0;i<16;i++)o.push(t().toString(16));return o.join("")}e.exports={generateUuid:r,generateCatId:o}},{}],22:[function(t,e,n){function r(t,e){if(!o)return!1;if(t!==o)return!1;if(!e)return!0;if(!i)return!1;for(var n=i.split("."),r=e.split("."),a=0;a<r.length;a++)if(r[a]!==n[a])return!1;return!0}var o=null,i=null,a=/Version\\/(\\S+)\\s+Safari/;if(navigator.userAgent){var s=navigator.userAgent,c=s.match(a);c&&s.indexOf("Chrome")===-1&&s.indexOf("Chromium")===-1&&(o="Safari",i=c[1])}e.exports={agent:o,version:i,match:r}},{}],23:[function(t,e,n){function r(t,e){var n=[],r="",i=0;for(r in t)o.call(t,r)&&(n[i]=e(r,t[r]),i+=1);return n}var o=Object.prototype.hasOwnProperty;e.exports=r},{}],24:[function(t,e,n){function r(t,e,n){e||(e=0),"undefined"==typeof n&&(n=t?t.length:0);for(var r=-1,o=n-e||0,i=Array(o<0?0:o);++r<o;)i[r]=t[e+r];return i}e.exports=r},{}],25:[function(t,e,n){e.exports={exists:"undefined"!=typeof window.performance&&window.performance.timing&&"undefined"!=typeof window.performance.timing.navigationStart}},{}],26:[function(t,e,n){function r(t){return!(t&&t instanceof Function&&t.apply&&!t[a])}var o=t("ee"),i=t(24),a="nr@original",s=Object.prototype.hasOwnProperty,c=!1;e.exports=function(t,e){function n(t,e,n,o){function nrWrapper(){var r,a,s,c;try{a=this,r=i(arguments),s="function"==typeof n?n(r,a):n||{}}catch(f){p([f,"",[r,a,o],s])}u(e+"start",[r,a,o],s);try{return c=t.apply(a,r)}catch(d){throw u(e+"err",[r,a,d],s),d}finally{u(e+"end",[r,a,c],s)}}return r(t)?t:(e||(e=""),nrWrapper[a]=t,d(t,nrWrapper),nrWrapper)}function f(t,e,o,i){o||(o="");var a,s,c,f="-"===o.charAt(0);for(c=0;c<e.length;c++)s=e[c],a=t[s],r(a)||(t[s]=n(a,f?s+o:o,i,s))}function u(n,r,o){if(!c||e){var i=c;c=!0;try{t.emit(n,r,o,e)}catch(a){p([a,n,r,o])}c=i}}function d(t,e){if(Object.defineProperty&&Object.keys)try{var n=Object.keys(t);return n.forEach(function(n){Object.defineProperty(e,n,{get:function(){return t[n]},set:function(e){return t[n]=e,e}})}),e}catch(r){p([r])}for(var o in t)s.call(t,o)&&(e[o]=t[o]);return e}function p(e){try{t.emit("internal-error",e)}catch(n){}}return t||(t=o),n.inPlace=f,n.flag=a,n}},{}],ee:[function(t,e,n){function r(){}function o(t){function e(t){return t&&t instanceof r?t:t?c(t,s,i):i()}function n(n,r,o,i){if(!p.aborted||i){t&&t(n,r,o);for(var a=e(o),s=m(n),c=s.length,f=0;f<c;f++)s[f].apply(a,r);var d=u[x[n]];return d&&d.push([g,n,r,a]),a}}function l(t,e){y[t]=m(t).concat(e)}function h(t,e){var n=y[t];if(n)for(var r=0;r<n.length;r++)n[r]===e&&n.splice(r,1)}function m(t){return y[t]||[]}function v(t){return d[t]=d[t]||o(n)}function w(t,e){f(t,function(t,n){e=e||"feature",x[n]=e,e in u||(u[e]=[])})}var y={},x={},g={on:l,addEventListener:l,removeEventListener:h,emit:n,get:v,listeners:m,context:e,buffer:w,abort:a,aborted:!1};return g}function i(){return new r}function a(){(u.api||u.feature)&&(p.aborted=!0,u=p.backlog={})}var s="nr@context",c=t("gos"),f=t(23),u={},d={},p=e.exports=o();p.backlog=u},{}],gos:[function(t,e,n){function r(t,e,n){if(o.call(t,e))return t[e];var r=n();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(t,e,{value:r,writable:!0,enumerable:!1}),r}catch(i){}return t[e]=r,r}var o=Object.prototype.hasOwnProperty;e.exports=r},{}],handle:[function(t,e,n){function r(t,e,n,r){o.buffer([t],r),o.emit(t,e,n)}var o=t("ee").get("handle");e.exports=r,r.ee=o},{}],id:[function(t,e,n){function r(t){var e=typeof t;return!t||"object"!==e&&"function"!==e?-1:t===window?0:a(t,i,function(){return o++})}var o=1,i="nr@id",a=t("gos");e.exports=r},{}],loader:[function(t,e,n){function r(){if(!E++){var t=b.info=NREUM.info,e=l.getElementsByTagName("script")[0];if(setTimeout(u.abort,3e4),!(t&&t.licenseKey&&t.applicationID&&e))return u.abort();f(x,function(e,n){t[e]||(t[e]=n)}),c("mark",["onload",a()+b.offset],null,"api");var n=l.createElement("script");n.src="https://"+t.agent,e.parentNode.insertBefore(n,e)}}function o(){"complete"===l.readyState&&i()}function i(){c("mark",["domContent",a()+b.offset],null,"api")}function a(){return R.exists&&performance.now?Math.round(performance.now()):(s=Math.max((new Date).getTime(),s))-b.offset}var s=(new Date).getTime(),c=t("handle"),f=t(23),u=t("ee"),d=t(22),p=window,l=p.document,h="addEventListener",m="attachEvent",v=p.XMLHttpRequest,w=v&&v.prototype;NREUM.o={ST:setTimeout,SI:p.setImmediate,CT:clearTimeout,XHR:v,REQ:p.Request,EV:p.Event,PR:p.Promise,MO:p.MutationObserver};var y=""+location,x={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net",agent:"js-agent.newrelic.com/nr-spa-1130.min.js"},g=v&&w&&w[h]&&!/CriOS/.test(navigator.userAgent),b=e.exports={offset:s,now:a,origin:y,features:{},xhrWrappable:g,userAgent:d};t(18),l[h]?(l[h]("DOMContentLoaded",i,!1),p[h]("load",r,!1)):(l[m]("onreadystatechange",o),p[m]("onload",r)),c("mark",["firstbyte",s],null,"api");var E=0,R=t(25)},{}]},{},["loader",2,16,5,3,4]);</script><script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","queueTime":0,"licenseKey":"6d0ece54b7","agent":"","transactionName":"NFBbY0VUXEEEARUPXg0af0JZVkZbCgxOBV4KW1RWRV5XRgYDEUhHClBORBlBQFMLEQ0HRQZRA1pWR1lXET0CB0E8V0BoVkNTWwkDAwpUPEZMR0dZS20EDg0=","applicationID":"217174376","errorBeacon":"bam.nr-data.net","applicationTime":11}</script>\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Cryptocurrency Market Capitalizations | CoinMarketCap</title>\n<link rel="preload" as="font" href="https://s2.coinmarketcap.com/static/cloud/fonts/glyphicons-regular.woff2" type="font/woff2" crossorigin>\n<link rel="preconnect" href="https://s2.coinmarketcap.com" crossorigin>\n<link rel="preconnect" href="https://files.coinmarketcap.com">\n<link rel="preconnect" href="//www.googletagmanager.com">\n<meta name="google-site-verification" content="EDc1reqlQ-zAgeRrrgAxRXNK-Zs9JgpE9a0wdaoSO9A">\n<meta property="og:type" content="website">\n<meta property="og:site_name" content="CoinMarketCap">\n<meta property="og:image" content="https://s2.coinmarketcap.com/static/cloud/img/splash_600x315_1.png">\n<meta property="og:image:type" content="image/png">\n<meta property="og:image:width" content="600">\n<meta property="og:image:height" content="315">\n<meta name="twitter:card" content="summary_large_image">\n<meta property="og:title" content="Cryptocurrency Market Capitalizations | CoinMarketCap" />\n<meta name="description" content="Cryptocurrency market cap rankings, charts, and more" />\n<meta property="og:description" content="Cryptocurrency market cap rankings, charts, and more" />\n<link rel="apple-touch-icon" href="/apple-touch-icon.png">\n<link rel="canonical" href="https://coinmarketcap.com/" />\n<meta property="og:url" content="https://coinmarketcap.com/">\n<link rel="alternate" href="https://coinmarketcap.com/de/" hreflang="de">\n<link rel="alternate" href="https://coinmarketcap.com/" hreflang="en">\n<link rel="alternate" href="https://coinmarketcap.com/es/" hreflang="es">\n<link rel="alternate" href="https://coinmarketcap.com/fil/" hreflang="fil">\n<link rel="alternate" href="https://coinmarketcap.com/fr/" hreflang="fr">\n<link rel="alternate" href="https://coinmarketcap.com/hi/" hreflang="hi">\n<link rel="alternate" href="https://coinmarketcap.com/it/" hreflang="it">\n<link rel="alternate" href="https://coinmarketcap.com/ja/" hreflang="ja">\n<link rel="alternate" href="https://coinmarketcap.com/ko/" hreflang="ko">\n<link rel="alternate" href="https://coinmarketcap.com/pt-br/" hreflang="pt-br">\n<link rel="alternate" href="https://coinmarketcap.com/ru/" hreflang="ru">\n<link rel="alternate" href="https://coinmarketcap.com/tr/" hreflang="tr">\n<link rel="alternate" href="https://coinmarketcap.com/vi/" hreflang="vi">\n<link rel="alternate" href="https://coinmarketcap.com/zh/" hreflang="zh">\n<link rel="alternate" href="https://coinmarketcap.com/zh-tw/" hreflang="zh-tw">\n<meta name="apple-itunes-app" content="app-id=1282107098">\n<link rel="manifest" href="/manifest.json">\n<link href="https://s2.coinmarketcap.com/static/cloud/compressed/base.52de1091.min.css" rel=\'stylesheet\'>\n<link href="https://files.coinmarketcap.com/static/header_banner/header-banner-production.css" rel=\'stylesheet\'>\n<script>\n        function getCookie(e){var n=" "+document.cookie,t=" "+e+"=",i=null,o=0,u=0;return n.length>0&&-1!=(o=n.indexOf(t))&&(o+=t.length,-1==(u=n.indexOf(";",o))&&(u=n.length),i=unescape(n.substring(o,u))),i}\n\n        var DEFAULT_CURRENCY = \'USD\';\n        dataLayer = [{\n            \'section\': \'Market Cap\',\n            \'subSection\': \'Cryptocurrencies\',\n            \'firstSessionDate\': getCookie(\'gtm_session_first\'),\n            \'lastSessionDate\': getCookie(\'gtm_session_last\'),\n            \'language\': \'en\',\n            \'currency\': getCookie(\'currency\') || DEFAULT_CURRENCY,\n            \'siteTheme\': getCookie(\'theme\') || \'day\',\n            \'siteNav\': \'v1\'\n        }];\n        </script>\n\n<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({\'gtm.start\': new Date().getTime(),event:\'gtm.js\'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!=\'dataLayer\'?\'&l=\'+l:\'\';j.async=true;j.src=\'//www.googletagmanager.com/gtm.js?id=\'+i+dl;f.parentNode.insertBefore(j,f);})(window,document,\'script\',\'dataLayer\',\'GTM-MNVXW26\');\n        </script>\n\n<script src="https://s2.coinmarketcap.com/static/cloud/compressed/basehead.cf08aec2.min.js"></script>\n<script async=\'async\' src="https://s2.coinmarketcap.com/static/cloud/compressed/prebid.dc802d7a.min.js"></script>\n<script async=\'async\' src=\'https://www.googletagservices.com/tag/js/gpt.js\'></script>\n<script>\n            var googletag = googletag || {};\n            googletag.cmd = googletag.cmd || [];\n            googletag.cmd.push(function() {\n                googletag.pubads().disableInitialLoad();\n            });\n        </script>\n<script>\n            var listingsAdSlots = [\n                \n                    {\n                        slotName: \'/48901027/listings_top\',\n                        sizes: [[320, 50], [728, 90], [320, 100]],\n                        slotID: \'div-gpt-ad-1542211140769-0\',\n                        mapping: [\n                          { browserSize: [768, 0], slotSizes: [[728, 90]] },\n                          { browserSize: [0, 0], slotSizes: [[320, 50], [320, 100]] }\n                        ]\n                    },\n                    {\n                        slotName: \'/48901027/listings_bottom\',\n                        sizes: [[320, 50], [728, 90], [320, 100]],\n                        slotID: \'div-gpt-ad-1542211140769-2\',\n                        mapping: [\n                          { browserSize: [768, 0], slotSizes: [[728, 90]] },\n                          { browserSize: [0, 0], slotSizes: [[320, 50], [320, 100]] }\n                        ]\n                    },\n                    {\n                        slotName: \'/48901027/listings_side\',\n                        sizes: [[160, 600]],\n                        slotID: \'div-gpt-ad-1542211140769-1\',\n                        mapping: [\n                          { browserSize: [1265, 0], slotSizes: [[160, 600]] },\n                          { browserSize: [0, 0], slotSizes: [] }\n                        ]\n                    }\n                \n            ];\n\n          var googleAdSlots = [];\n\n          googletag.cmd.push(function() {\n            for (var i = 0, slotLen = listingsAdSlots.length; i < slotLen; i++) {\n                var adSlot = listingsAdSlots[i];\n\n                var slotMapping = [];\n                for (var j = 0, mapLen = adSlot.mapping.length; j < mapLen; j++) {\n                      var mapping = adSlot.mapping[j];\n                      slotMapping.push([mapping.browserSize, mapping.slotSizes]);\n                }\n\n                slot = googletag.defineSlot(adSlot.slotName, adSlot.sizes[0], adSlot.slotID).\n                    defineSizeMapping(slotMapping).\n                    addService(googletag.pubads());\n                googleAdSlots.push(slot);\n            }\n\n            \n\n            \n\n            var mapping = googletag.sizeMapping()\n              .addSize([1280,0], [1280, 90])\n              .addSize([970,0], [970, 90])\n              .addSize([750,0], [750, 90])\n              .addSize([0,0], [320, 100])\n              .build();\n            var headerBannerSlot = googletag.defineSlot(\'/48901027/full_width_top\', [960, 90], \'div-gpt-ad-1517714727704-0\')\n                .defineSizeMapping(mapping)\n                .addService(googletag.pubads())\n                .setCollapseEmptyDiv(true);\n\n            googletag.pubads().addEventListener(\'slotRenderEnded\', function(event) {\n                if (event.slot === headerBannerSlot) {\n                    var display = document.getElementById(\'div-gpt-ad-1517714727704-0\').style.display,\n                        el = document.getElementById(\'header-banner-wrapper\');\n                    if (display === \'none\') {\n                        el.style.display = \'none\';\n                    } else {\n                        el.style.visibility = \'visible\';\n                    }\n                }\n            });\n\n            googletag.pubads().enableSingleRequest();\n            googletag.enableServices();\n          });\n        </script>\n<script>\n        //Load the UAM JavaScript Library\n        !function(a9,a,p,s,t,A,g){if(a[a9])return;function\n        q(c,r){a[a9]._Q.push([c,r])}a[a9]={init:function(){q("i",arguments)},fetchBids:function(){q("f",arguments)},setDisplayBids:function(){},targetingKeys:function(){return[]},_Q:[]};A=p.createElement(s);A.async=!0;A.src=t;g=p.getElementsByTagName(s)[0];g.parentNode.insertBefore(A,g)}("apstag",window,document,"script","//c.amazon-adsystem.com/aax2/apstag.js");\n\n        apstag.init({\n            pubID: \'1edde03c-121e-457d-9c35-ad4fca989bac\',\n            adServer: \'googletag\',\n            bidTimeout: 1200,\n            simplerGPT: true\n        });\n\n        var pbjs = pbjs || {};\n        pbjs.que = pbjs.que || [];\n\n        </script>\n<script>\n        function fetchHeaderBids() {\n            var bidTimeout = 1200;\n            var adUnits = [\n            \n            {\n                code: \'div-gpt-ad-1542211140769-0\', //Listings Top\n                mediaTypes: {\n                    banner: {\n                        sizes: [[320, 100], [728, 90], [320, 50]]\n                    }\n                },\n                bids: [\n                {\n                    "bidder": "appnexus",\n                    "params": {\n                        "placementId": "15436530", // [R] Listings Top\n                        "allowSmallerSizes": true\n                    }\n                },\n                {\n                    "bidder": "ix",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "size": [320,100],\n                        "siteId": "219070" // Listings Top\n                    }\n                },\n                {\n                    "bidder": "ix",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "size": [728, 90],\n                        "siteId": "219064" // Listings Top\n                    }\n                },\n                {\n                    "bidder": "ix",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "size": [320, 50],\n                        "siteId": "219070" // Listings Top\n                    }\n                },\n                {\n                    "bidder": "openx",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "unit": "539181147",\n                        "delDomain": "coinmarketcap-d.openx.net" // 728x90 Listings Top\n                    }\n                },\n                {\n                    "bidder": "openx",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "unit": "540631895",\n                        "delDomain": "coinmarketcap-d.openx.net" // 320x100 Listings Top\n                    }\n                },\n                {\n                    "bidder": "openx",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "unit": "539181154",\n                        "delDomain": "coinmarketcap-d.openx.net" // 320x50 Listings Top\n                    }\n                },\n                {\n                    "bidder": "pubmatic",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "publisherId": "157006",\n                        "adSlot": "1763907" // 728x90 - Listings Top\n                    }\n                },\n                {\n                    "bidder": "pubmatic",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "publisherId": "157006",\n                        "adSlot": "1763914" // 320x100 - Listings Top\n                    }\n                },\n                {\n                    "bidder": "pubmatic",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "publisherId": "157006",\n                        "adSlot": "1763918" // 320x50 - Listings Top\n                    }\n                },\n                {\n                    "bidder": "coinzilla",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "placementId": "8125c7f735236e75384", // 728x90\n                    }\n                },\n                {\n                    "bidder": "coinzilla",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "placementId": "2555c7f73b8734cb295", // 320x100\n                    }\n                },\n                {\n                    "bidder": "sovrn",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "tagid": "496628" // CoinMarketCap_AiH_728x90_ATF_HOMEPAGE_Desktop\n                    }\n                },\n                {\n                    "bidder": "sovrn",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "tagid": "499686" // CoinMarketCap_AiH_320x100_ATF_HOMEPAGE_Mobile\n                    }\n                },\n                {\n                    "bidder": "sovrn",\n                    "labelAny": ["mobile"],\n                    "params": {\n                        "tagid": "496634" // CoinMarketCap_AiH_320x50_ATF_HOMEPAGE_Mobile\n                    }\n                }\n                ]\n            },{\n                code: \'div-gpt-ad-1542211140769-1\', // Listings Side\n                mediaTypes: {\n                    banner: {\n                        sizes: [[160, 600]]\n                    }\n                },\n                labelAny: ["desktop-wide"],\n                bids: [\n                {\n                    "bidder": "appnexus",\n                    "params": {\n                        "placementId": "15436531", // [R] Listings Side\n                        "allowSmallerSizes": true\n                    }\n                },\n                {\n                    "bidder": "ix",\n                    "params": {\n                        "size": [160, 600],\n                        "siteId": "219065" // Listings Side\n                    }\n                },\n                {\n                    "bidder": "openx",\n                    "params": {\n                        "unit": "539181151",\n                        "delDomain": "coinmarketcap-d.openx.net" // 160x600 Listings\n                    }\n                },\n                {\n                    "bidder": "pubmatic",\n                    "params": {\n                        "publisherId": "157006",\n                        "adSlot": "1763908" // 160x600 - Listings\n                    }\n                },\n                {\n                    "bidder": "coinzilla",\n                    "params": {\n                        "placementId": "6175c7f73523aeec252", // 160x60\n                    }\n                },\n                {\n                    "bidder": "sovrn",\n                    "params": {\n                        "tagid": "496632" // CoinMarketCap_AiH_160x600_Side_HOMEPAGE_Desktop\n                    }\n                }\n                ]\n            },{\n                code: \'div-gpt-ad-1542211140769-2\', // Listings Bottom\n                mediaTypes: {\n                    banner: {\n                        sizes: [[728, 90]]\n                    }\n                },\n                bids: [\n                {\n                    "bidder": "appnexus",\n                    "params": {\n                        "placementId": "15436532", // [R] Listings Bottom\n                        "allowSmallerSizes": true\n                    }\n                },\n                {\n                    "bidder": "ix",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "size": [728, 90],\n                        "siteId": "219066" // Listings Bottom\n                    }\n                },\n                {\n                    "bidder": "pubmatic",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "publisherId": "157006",\n                        "adSlot": "1763909" // 728x90 - Listings Bottom\n                    }\n                },\n                {\n                    "bidder": "sovrn",\n                    "labelAny": ["desktop"],\n                    "params": {\n                        "tagid": "496630" // CoinMarketCap_AiH_728x90_BTF_HOMEPAGE_Desktop\n                    }\n                }\n                ]\n            }\n            \n            ];\n\n            var apstagSlots = googleAdSlots;\n\n            var bidders = [\'a9\', \'prebid\'];\n            var requestManager = {\n                adserverRequestSent: false,\n            };\n\n            bidders.forEach(function(bidder) {\n                requestManager[bidder] = false;\n            })\n\n            function allBiddersBack() {\n                var allBiddersBack = bidders\n                .map(function(bidder) {\n                    return requestManager[bidder];\n                })\n\n                .filter(Boolean)\n                .length === bidders.length;\n                return allBiddersBack;\n            }\n\n            function headerBidderBack(bidder) {\n                if (requestManager.adserverRequestSent === true) {\n                    return;\n                }\n                if (bidder === \'a9\') {\n                    requestManager.a9 = true;\n                } else if (bidder === \'prebid\') {\n                    requestManager.prebid = true;\n                }\n\n                if (allBiddersBack()) {\n                    sendAdserverRequest();\n                }\n            }\n\n            function sendAdserverRequest() {\n                if (requestManager.adserverRequestSent === true) {\n                    return;\n                }\n                requestManager.adserverRequestSent = true;\n                pbjs.adserverRequestSent = true;\n                requestManager.sendAdserverRequest = true;\n                googletag.cmd.push(function() {\n                    apstag.setDisplayBids();\n                    if (pbjs.version) {\n                        pbjs.setTargetingForGPTAsync();\n                    }\n                    googletag.pubads().refresh();\n                });\n            }\n\n            function requestBids(apstagSlots, adUnits, bidTimeout) {\n                apstag.fetchBids({\n                    slots: apstagSlots,\n                    timeout: bidTimeout\n                    }, function(bids) {\n                        headerBidderBack(\'a9\');\n                    });\n\n                pbjs.que.push(function() {\n                    pbjs.setConfig({\n                        enableSendAllBids: true,\n                        priceGranularity: "high",\n                        userSync: {\n                            filterSettings: {\n                                iframe: {\n                                    bidders: \'*\',\n                                    filter: \'include\'\n                                }\n                            }\n                        },\n                        sizeConfig: [{\n                            \'mediaQuery\': \'(min-width: 768px)\',\n                            \'sizesSupported\': [[728,90]],\n                            \'labels\': [\'desktop\']\n                        }, {\n                            \'mediaQuery\': \'(min-width: 1265px)\',\n                            \'sizesSupported\': [[160,600]],\n                            \'labels\': [\'desktop-wide\']\n                        }, {\n                            \'mediaQuery\': \'(max-width: 767px)\',\n                            \'sizesSupported\': [[320,100],[320,50],[300,250]],\n                            \'labels\': [\'mobile\']\n                        }]\n\n                    });\n                    pbjs.bidderSettings = {\n                        openx: {\n                            bidCpmAdjustment : function(bidCpm) {return bidCpm * 0.6;}\n                        },\n                        sovrn: {\n                            bidCpmAdjustment : function(bidCpm) {return bidCpm * 0.7;}\n                        },\n                        appnexus: {\n                            bidCpmAdjustment : function(bidCpm) {return bidCpm * 0.75;}\n                        },\n                        coinzilla: {\n                            bidCpmAdjustment : function(bidCpm) {return bidCpm * 0.99;}\n                        },\n                        pubmatic: {\n                            bidCpmAdjustment : function(bidCpm) {return bidCpm * 0.7;}\n                        }\n                    };\n                    pbjs.addAdUnits(adUnits);\n                    pbjs.requestBids({\n                        bidsBackHandler: function (bids) {\n                            headerBidderBack(\'prebid\');\n                        }\n                    });\n                });\n            }\n\n            requestBids(apstagSlots, adUnits, bidTimeout)\n\n            window.setTimeout(function() {\n                sendAdserverRequest();\n            }, bidTimeout);\n        };\n\n        googletag.cmd.push(function() {\n            fetchHeaderBids();\n        });\n        </script>\n<script type="application/ld+json">\n            {\n                "@context": "http://schema.org",\n                "@type": "Organization",\n                "name": "CoinMarketCap",\n                "url": "https://coinmarketcap.com",\n                "logo": "https://coinmarketcap.com/apple-touch-icon.png",\n                "sameAs": [\n                    "https://www.facebook.com/CoinMarketCap/",\n                    "https://twitter.com/coinmarketcap"\n                ]\n            }\n        </script>\n<script type="application/ld+json">\n            {\n                "@context": "http://schema.org",\n                "@type": "Table",\n                "about": "Cryptocurrency Prices Today"\n            }\n        </script>\n<script>\n        window.lazySizesConfig = {\n            expFactor: 4\n        };\n    </script>\n<script src="https://s2.coinmarketcap.com/static/cloud/compressed/currencies_top.06f5ed29.min.js"></script>\n</head>\n<body>\n\n<noscript><iframe src="//www.googletagmanager.com/ns.html?id=GTM-MNVXW26" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>\n\n<!--[if lt IE 7]>\n            <p class="chromeframe">You are using an <strong>outdated</strong> browser. Please <a href="http://browsehappy.com/">upgrade your browser</a> or <a href="http://www.google.com/chromeframe/?redirect=true">activate Google Chrome Frame</a> to improve your experience.</p>\n        <![endif]-->\n<script>\n            NightMode.init();\n        </script>\n<div class="cmc-nav-wrap">\n<div id="header-banner-wrapper">\n\n<div id="div-gpt-ad-1517714727704-0" class="centered">\n<script>\n                    googletag.cmd.push(function() { googletag.display(\'div-gpt-ad-1517714727704-0\'); });\n                </script>\n</div>\n<div class="header-banner-close banner-alert-close pointer">&times;</div>\n</div>\n<script>\n                if (getCookie("header-banner-noshow") === "1") {\n                    document.getElementById(\'header-banner-wrapper\').style.display = \'none\';\n                }\n            </script>\n<div class="cmc-nav">\n<div class="cmc-nav__topbar cmc-nav-desktop">\n<div class="container">\n<div class="cmc-global-stats visibility-hidden js-global-stats">\n<span>\nCryptocurrencies:&nbsp;\n<a href="/all/views/all/"><span class="js-global-stats-cryptocurrencies"></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nMarkets:&nbsp;\n<a href="/currencies/volume/24-hour/"><span class="js-global-stats-markets"></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nMarket Cap:&nbsp;\n<a href="/charts/"><span class="js-global-stats-market-cap" data-global-currency-market-cap></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\n24h Vol:&nbsp;\n<a href="/charts/"><span class="js-global-stats-volume" data-global-currency-volume></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nBTC Dominance:&nbsp;\n<a href="/charts/#dominance-percentage"><span class="js-global-stats-btc-dominance"></span>%</a>\n</span>\n</div>\n<div class="cmc-siteprefs">\n<a class="cmc-link js-nav-track-click" href="https://coinmarketcap.com/advertising" target="_blank" rel="noopener" data-linkid="Advertise - topbar">Advertise</a>\n<div class="cmc-siteprefs-separator"></div>\n<div class="cmc-siteprefs-control cmc-siteprefs-dropdown js-nav-lang-picker">\n<button type="button" class="cmc-nav-button dropdown-toggle language-dropdown" data-toggle="dropdown" title="Change your language">\n<span>EN</span>&nbsp;<span class="caret"></span>\n</button>\n<ul class="dropdown-menu cmc-siteprefs-dropdown-menu cmc-siteprefs-lang-picker__menu" role="menu">\n<li class="pointer"><a href="/de/" data-language-toggle="de">Deutsch</a></li>\n<li class="pointer"><a href="/" data-language-toggle="en">English</a></li>\n<li class="pointer"><a href="/es/" data-language-toggle="es">Espa\xc3\xb1ol</a></li>\n<li class="pointer"><a href="/fil/" data-language-toggle="fil">Filipino</a></li>\n<li class="pointer"><a href="/fr/" data-language-toggle="fr">Fran\xc3\xa7ais</a></li>\n<li class="pointer"><a href="/hi/" data-language-toggle="hi">\xe0\xa4\xb9\xe0\xa4\xbf\xe0\xa4\xa8\xe0\xa5\x8d\xe0\xa4\xa6\xe0\xa5\x80</a></li>\n<li class="pointer"><a href="/it/" data-language-toggle="it">Italiano</a></li>\n<li class="pointer"><a href="/ja/" data-language-toggle="ja">\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e</a></li>\n<li class="pointer"><a href="/ko/" data-language-toggle="ko">\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4</a></li>\n<li class="pointer"><a href="/pt-br/" data-language-toggle="pt-br">Portugu\xc3\xaas Brasil</a></li>\n<li class="pointer"><a href="/ru/" data-language-toggle="ru">\xd0\xa0\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9</a></li>\n<li class="pointer"><a href="/tr/" data-language-toggle="tr">T\xc3\xbcrk\xc3\xa7e</a></li>\n<li class="pointer"><a href="/vi/" data-language-toggle="vi">Ti\xe1\xba\xbfng Vi\xe1\xbb\x87t</a></li>\n<li class="pointer"><a href="/zh/" data-language-toggle="zh">\xe7\xae\x80\xe4\xbd\x93\xe4\xb8\xad\xe6\x96\x87</a></li>\n<li class="pointer"><a href="/zh-tw/" data-language-toggle="zh-tw">\xe7\xb9\x81\xe9\xab\x94\xe4\xb8\xad\xe6\x96\x87</a></li>\n</ul>\n</div>\n<div data-global-currency-switch class="cmc-siteprefs-control cmc-siteprefs-dropdown global-currency-dropdown-container">\n<button type="button" class="cmc-nav-button dropdown-toggle global-currency-dropdown" data-toggle="dropdown" title="Change your display currency">\n<span data-currency-flag-display></span>&nbsp;<span class="caret"></span>\n</button>\n<ul class="dropdown-menu cmc-siteprefs-dropdown-menu cmc-siteprefs-currency-picker__menu" role="menu">\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="USD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-us">\n</span>\n<span>USD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="AUD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-au">\n</span>\n<span>AUD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="BRL">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-br">\n</span>\n<span>BRL</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CAD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ca">\n</span>\n<span>CAD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CHF">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ch">\n</span>\n<span>CHF</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CLP">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cl">\n</span>\n<span>CLP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CNY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cn">\n</span>\n <span>CNY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CZK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cz">\n</span>\n<span>CZK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="DKK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-dk">\n</span>\n<span>DKK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="EUR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-eu">\n</span>\n<span>EUR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="GBP">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-gb">\n</span>\n<span>GBP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="HKD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-hk">\n</span>\n<span>HKD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="HUF">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-hu">\n</span>\n<span>HUF</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="IDR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-id">\n</span>\n<span>IDR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="ILS">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-il">\n</span>\n<span>ILS</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="INR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-in">\n</span>\n<span>INR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="JPY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-jp">\n</span>\n<span>JPY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="KRW">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-kr">\n</span>\n<span>KRW</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="MXN">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-mx">\n</span>\n<span>MXN</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="MYR">\n <span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-my">\n</span>\n<span>MYR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="NOK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-no">\n</span>\n<span>NOK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="NZD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-nz">\n</span>\n<span>NZD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PHP">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ph">\n</span>\n<span>PHP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PKR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-pk">\n</span>\n<span>PKR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PLN">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-pl">\n</span>\n<span>PLN</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="RUB">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ru">\n</span>\n<span>RUB</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="SEK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-se">\n</span>\n<span>SEK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="SGD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-sg">\n</span>\n<span>SGD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="THB">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-th">\n</span>\n<span>THB</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="TRY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-tr">\n</span>\n<span>TRY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="TWD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-tw">\n</span>\n<span>TWD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="ZAR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-za">\n</span>\n<span>ZAR</span>\n</li>\n</ul>\n </div>\n<div class="cmc-siteprefs-control">\n<button type="button" class="cmc-nav-button js-theme-switch" title="Day/Night Mode"><span class="icon-moon"></span></button>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-nav-mobile">\n<nav class="cmc-nav-mobile-navbar">\n<div class="container">\n<div class="cmc-nav-mobile-hamburger-menu js-nav-mobile-takeover-opener">\n<span class="icon-bars"></span>\n</div>\n<div class="cmc-nav-mobile-logo">\n<a href="/" aria-label="Go to the homepage">\n<span class="cmc-logo cmc-logo_color_black cmc-navbar__logo"></span>\n</a>\n</div>\n<div class="cmc-search-mobile js-search-mobile-opener">\n<span class="icon-search"></span>\n</div>\n</div>\n</nav>\n<div class="cmc-nav-mobile-takeover cmc-nav-mobile-takeover-enter js-nav-mobile-takeover">\n<div class="cmc-nav-mobile-takeover__header">\n<div class="cmc-nav-mobile-takeover__logo">\n<a href="/" aria-label="Go to the homepage">\n<span class="cmc-logo cmc-logo_color_black cmc-navbar__logo"></span>\n</a>\n</div>\n<div class="cmc-nav-mobile-close js-nav-mobile-takeover-closer">\n<span class="cmc-nav-mobile-close-icon icon-times"></span>\n</div>\n</div>\n<ul class="cmc-nav-mobile-takeover__menulist" role="menu">\n<li class="cmc-nav-mobile-takeover__menuitem js-nav-mobile-submenu-open" data-menu-id="rankings" role="menuitem">\n<div>Rankings</div>\n<div class=""><span class="icon-angle-right"></span></div>\n</li>\n<li class="cmc-nav-mobile-takeover__menuitem js-nav-mobile-submenu-open" data-menu-id="tools" role="menuitem">\n<div>Tools</div>\n<div class=""><span class="icon-angle-right"></span></div>\n</li>\n<li class="cmc-nav-mobile-takeover__menuitem js-nav-mobile-submenu-open" data-menu-id="resources" role="menuitem">\n<div>Resources</div>\n<div class=""><span class="icon-angle-right"></span></div>\n</li>\n<li class="cmc-nav-mobile-takeover__menuitem js-nav-mobile-submenu-open" data-menu-id="more" role="menuitem">\n<div>Helpful Links</div>\n<div class=""><span class="icon-angle-right"></span></div>\n</li>\n<li class="cmc-nav-mobile-takeover__menuitem" role="menuitem">\n<a href="https://blog.coinmarketcap.com" target="_blank" rel="noopener">Blog</a>\n</li>\n</ul>\n<div class="cmc-nav-mobile-siteprefs">\n<div class="cmc-nav-mobile-siteprefs-control cmc-siteprefs-dropdown js-nav-lang-picker">\n<button type="button" class="cmc-nav-mobile-button dropdown-toggle language-dropdown" data-toggle="dropdown" title="Change your language">\n<span>EN</span>&nbsp;<span class="caret"></span>\n</button>\n<ul class="dropdown-menu cmc-nav-mobile-siteprefs-dropdown-menu cmc-siteprefs-lang-picker__menu" role="menu">\n<li><a href="/de/" data-language-toggle="de">Deutsch</a></li>\n<li><a href="/" data-language-toggle="en">English</a></li>\n<li><a href="/es/" data-language-toggle="es">Espa\xc3\xb1ol</a></li>\n<li><a href="/fil/" data-language-toggle="fil">Filipino</a></li>\n<li><a href="/fr/" data-language-toggle="fr">Fran\xc3\xa7ais</a></li>\n<li><a href="/hi/" data-language-toggle="hi">\xe0\xa4\xb9\xe0\xa4\xbf\xe0\xa4\xa8\xe0\xa5\x8d\xe0\xa4\xa6\xe0\xa5\x80</a></li>\n<li><a href="/it/" data-language-toggle="it">Italiano</a></li>\n<li><a href="/ja/" data-language-toggle="ja">\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e</a></li>\n<li><a href="/ko/" data-language-toggle="ko">\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4</a></li>\n<li><a href="/pt-br/" data-language-toggle="pt-br">Portugu\xc3\xaas Brasil</a></li>\n<li><a href="/ru/" data-language-toggle="ru">\xd0\xa0\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9</a></li>\n<li><a href="/tr/" data-language-toggle="tr">T\xc3\xbcrk\xc3\xa7e</a></li>\n<li><a href="/vi/" data-language-toggle="vi">Ti\xe1\xba\xbfng Vi\xe1\xbb\x87t</a></li>\n<li><a href="/zh/" data-language-toggle="zh">\xe7\xae\x80\xe4\xbd\x93\xe4\xb8\xad\xe6\x96\x87</a></li>\n<li><a href="/zh-tw/" data-language-toggle="zh-tw">\xe7\xb9\x81\xe9\xab\x94\xe4\xb8\xad\xe6\x96\x87</a></li>\n</ul>\n</div>\n<div data-global-currency-switch class="cmc-nav-mobile-siteprefs-control cmc-siteprefs-dropdown global-currency-dropdown-container">\n<button type="button" class="cmc-nav-mobile-button dropdown-toggle global-currency-dropdown" data-toggle="dropdown" title="Change your display currency">\n<span data-currency-flag-display></span>&nbsp;<span class="caret"></span>\n</button>\n<ul class="dropdown-menu cmc-nav-mobile-siteprefs-dropdown-menu cmc-siteprefs-currency-picker__menu" role="menu">\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="USD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-us">\n</span>\n<span>USD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="AUD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-au">\n</span>\n<span>AUD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="BRL">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-br">\n</span>\n<span>BRL</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CAD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ca">\n</span>\n<span>CAD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CHF">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ch">\n</span>\n<span>CHF</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CLP">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cl">\n</span>\n<span>CLP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CNY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cn">\n</span>\n<span>CNY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="CZK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-cz">\n</span>\n<span>CZK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="DKK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-dk">\n</span>\n<span>DKK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="EUR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-eu">\n</span>\n<span>EUR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="GBP">\n <span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-gb">\n</span>\n<span>GBP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="HKD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-hk">\n</span>\n<span>HKD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="HUF">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-hu">\n</span>\n<span>HUF</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="IDR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-id">\n</span>\n<span>IDR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="ILS">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-il">\n</span>\n<span>ILS</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="INR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-in">\n</span>\n<span>INR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="JPY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-jp">\n</span>\n<span>JPY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="KRW">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-kr">\n</span>\n<span>KRW</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="MXN">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-mx">\n</span>\n<span>MXN</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="MYR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-my">\n</span>\n<span>MYR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="NOK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-no">\n</span>\n<span>NOK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="NZD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-nz">\n</span>\n<span>NZD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PHP">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ph">\n</span>\n<span>PHP</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PKR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-pk">\n</span>\n<span>PKR</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="PLN">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-pl">\n</span>\n<span>PLN</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="RUB">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-ru">\n</span>\n<span>RUB</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="SEK">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-se">\n</span>\n<span>SEK</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="SGD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-sg">\n</span>\n<span>SGD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="THB">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-th">\n</span>\n<span>THB</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="TRY">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-tr">\n</span>\n<span>TRY</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="TWD">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-tw">\n</span>\n<span>TWD</span>\n</li>\n<li class="cmc-siteprefs-dropdown-menuitem" data-currency-flag-toggle data-currency-code="ZAR">\n<span class="cmc-siteprefs__currency-flag cmc-flag cmc-flag-za">\n</span>\n<span>ZAR</span>\n</li>\n</ul>\n</div>\n<div class="cmc-nav-mobile-siteprefs-control cmc-nav-mobile-siteprefs-control--theme">\n<button type="button" class="cmc-nav-mobile-button js-theme-switch" title="Day/Night Mode"><span class="icon-moon"></span></button>\n</div>\n</div>\n<div class="cmc-nav-mobile-takeover__footer">\n<div>\n<a class="cmc-link-secondary" href="/disclaimer/">Disclaimer</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary" href="/request/" target="_blank">Request Form</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary" href="/terms/">Terms of Use</a>\n</div>\n<div>\n<a class="cmc-link-secondary" href="/privacy/">Privacy Policy</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary js-nav-track-click" href="/advertising/" target="_blank" data-linkid="Advertise - mobile footer">Advertise</a>\n</div>\n<hr />\n<ul class="cmc-nav-mobile-social-links cmc-list-inline">\n<li><a href="https://twitter.com/CoinMarketCap" target="_blank" rel="noopener" title="Twitter"><span class="icon-twitter cmc-nav-mobile-social-icon"></span></a></li>\n<li><a href="https://www.facebook.com/CoinMarketCap" target="_blank" rel="noopener" title="Facebook"><span class="icon-facebook cmc-nav-mobile-social-icon"></span></a></li>\n<li><a href="https://t.me/CoinMarketCap" target="_blank" rel="nofollow noopener" title="Telegram"><span class="icon-telegram cmc-nav-mobile-social-icon"></span></a></li>\n<li><a href="https://www.linkedin.com/company/coinmarketcap/" target="_blank" rel="nofollow noopener" title="LinkedIn"><span class="icon-linkedin cmc-nav-mobile-social-icon"></span></a></li>\n<li><a href="https://www.instagram.com/coinmarketcap/" target="_blank" rel="nofollow noopener" title="Instagram"><span class="icon-instagram cmc-nav-mobile-social-icon"></span></a></li>\n<li><a href="https://www.reddit.com/r/CoinMarketCap" target="_blank" rel="nofollow noopener" title="Reddit"><span class="icon-reddit cmc-nav-mobile-social-icon"></span></a></li>\n</ul>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu-takeover cmc-nav-mobile-submenu-enter js-nav-mobile-submenu-takeover">\n<div class="cmc-nav-mobile-submenu__header">\n<button class="u-button-reset cmc-nav-mobile-submenu__back-button js-nav-mobile-submenu__back-button">\n<span class="icon-angle-left"></span> Back to Main Menu\n</button>\n<button class="u-button-reset cmc-nav-mobile-close js-nav-mobile-submenu-closer" title="Exit the navigation menu">\n<span class="cmc-nav-mobile-close-icon icon-times"></span>\n</button>\n</div>\n<div class="cmc-nav-mobile-submenu is-closed js-nav-dropdown-menu" data-menu-id="rankings">\n<div class="cmc-nav-mobile-submenu__title">\nRankings\n</div>\n<div class="cmc-nav-mobile-submenu__section cmc-nav-mobile-submenu__section--layout-two-col">\n<div class="cmc-nav-mobile-submenu__item">\n<h6 class="cmc-nav-mobile-heading">Market Cap</h6>\n<ul class="cmc-nav-mobile-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/">All Cryptocurrencies</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/coins/">Coins</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/tokens/">Tokens</a></li>\n</ul>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<h6 class="cmc-nav-mobile-heading">Trends</h6>\n<ul class="cmc-nav-mobile-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/gainers-losers/">Gainers &amp; Losers</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/new/">Recently Added</a></li>\n</ul>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<h6 class="cmc-nav-mobile-heading">Cryptocurrencies</h6>\n<ul class="cmc-nav-mobile-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/currencies/volume/24-hour/">Daily Volume</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/currencies/volume/monthly/">Monthly Volume</a></li>\n</ul>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<h6 class="cmc-nav-mobile-heading">Exchanges</h6>\n<ul class="cmc-nav-mobile-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/rankings/exchanges/">Adjusted Volume</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/rankings/exchanges/reported/">Reported Volume</a></li>\n</ul>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--according-to-cmc cmc-nav-spotlight--mobile">\n<a class="cmc-nav-spotlight__link js-nav-track-click" href="https://blog.coinmarketcap.com/2018/11/27/according-to-coinmarketcap-2018-edition/?utm_source=coinmarketcap.com&utm_medium=web&utm_campaign=nav2" target="_blank" rel="noopener" title="According to CoinMarketCap blog post" data-linkid="According to CMC - mobile"></a>\n</div>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu is-closed js-nav-dropdown-menu" data-menu-id="tools">\n<div class="cmc-nav-mobile-submenu__title">\nTools\n</div>\n <div class="cmc-nav-mobile-submenu__section cmc-nav-mobile-submenu__section--layout-two-col">\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/charts/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Global Charts</h6>\n<p class="cmc-nav-mobile-desc">Total Market Cap &amp; Dominance Charts</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/converter/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Currency Converter</h6>\n<p class="cmc-nav-mobile-desc">Convert prices of thousands of crypto and fiat currencies</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/historical/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Historical Snapshots</h6>\n<p class="cmc-nav-mobile-desc">View crypto rankings from the past</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/widget/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Site Widgets</h6>\n<p class="cmc-nav-mobile-desc">Powerful &amp; reliable widgets for your site</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/watchlist/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Watchlist</h6>\n<p class="cmc-nav-mobile-desc">Keep an eye on your favorite cryptocurrencies</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="https://blockchain.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Blockchain Explorer</h6>\n<p class="cmc-nav-mobile-desc">Search and visualize blockchain data</p>\n</a>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--tools cmc-nav-spotlight--mobile">\n<a class="cmc-nav-spotlight__link" href="https://coinmarketcap.com/mobile/" target="_blank"></a>\n</div>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu is-closed js-nav-dropdown-menu" data-menu-id="resources">\n<div class="cmc-nav-mobile-submenu__title">\nResources\n</div>\n<div class="cmc-nav-mobile-submenu__section cmc-nav-mobile-submenu__section--layout-two-col">\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/api/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Crypto API</h6>\n<p class="cmc-nav-mobile-desc">A powerful &amp; flexible market data API for commercial &amp; personal use</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/#newsletter" target="_blank">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Newsletter</h6>\n<p class="cmc-nav-mobile-desc">Articles &amp; interviews for the crypto &amp; blockchain community</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/events/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Events Calendar</h6>\n<p class="cmc-nav-mobile-desc">A full list of crypto &amp; blockchain events from around the world</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/indices/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Crypto Indices</h6>\n<p class="cmc-nav-mobile-desc">The most comprehensive suite of institutional grade indices in the market</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/glossary/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Crypto Glossary</h6>\n<p class="cmc-nav-mobile-desc">Learn the terms, slang and definitions around the crypto space</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/intro-to-crypto/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Intro to Crypto</h6>\n<p class="cmc-nav-mobile-desc">Learn about the world of cryptocurrencies with our introductory guide</p>\n</a>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--resources cmc-nav-spotlight--mobile">\n<a class="cmc-nav-spotlight__link" href="https://coinmarketcap.com/mobile/" target="_blank"></a>\n</div>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu is-closed js-nav-dropdown-menu" data-menu-id="more">\n<div class="cmc-nav-mobile-submenu__title">\nHelpful Links\n</div>\n<div class="cmc-nav-mobile-submenu__section cmc-nav-mobile-submenu__section--layout-two-col">\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/methodology/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Methodology</h6>\n<p class="cmc-nav-mobile-desc">How CoinMarketCap analyzes data to offer up-to-the-minute updates</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/careers/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Careers</h6>\n<p class="cmc-nav-mobile-desc">Discover your next career! Build the future of crypto data with us</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="/faq/">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">FAQ</h6>\n<p class="cmc-nav-mobile-desc">Need help? Check our FAQ, with questions we are most commonly asked</p>\n</a>\n</div>\n<div class="cmc-nav-mobile-submenu__item">\n<a class="cmc-nav-link--menuitem" href="https://shop.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">\n<h6 class="cmc-nav-mobile-heading cmc-nav-link--menuitem__arrow">Shop</h6>\n<p class="cmc-nav-mobile-desc">Purchase exclusive CoinMarketCap apparel!</p>\n</a>\n</div>\n</div>\n<div class="cmc-nav-mobile-submenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--intro-to-crypto cmc-nav-spotlight--mobile">\n<a class="cmc-nav-spotlight__link js-nav-track-click" href="/intro-to-crypto/" target="_blank" title="Intro to Crypto Guide" data-linkid="Intro to CMC - mobile"></a>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-search-mobile-takeover js-search-mobile-takeover">\n<div class="cmc-search-mobile-bar">\n<div class="cmc-quick-search cmc-quick-search--mobile">\n<form action="/search/" role="search">\n<input type="text" class="cmc-search-mobile__input js-quick-search js-search-mobile__input" placeholder="Search" name="q" aria-label="Search">\n</form>\n</div>\n<div class="cmc-search-mobile-closer js-search-mobile-closer">\n<button class="cmc-search-mobile-closer__button" type="button">Cancel</button>\n</div>\n</div>\n</div>\n<div class="cmc-nav-mobile__topbar js-global-stats-mobile__container">\n<div class="container cmc-global-stats__container">\n<div class="cmc-global-stats__fade"></div>\n<div class="cmc-global-stats__content">\n<div class="cmc-global-stats cmc-global-stats__inner-content visibility-hidden js-global-stats js-global-stats-mobile">\n<span>\nMarket Cap:&nbsp;\n<a href="/charts/"><span class="js-global-stats-market-cap" data-global-currency-market-cap></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\n24h Vol:&nbsp;\n<a href="/charts/"><span class="js-global-stats-volume" data-global-currency-volume></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nBTC Dominance:&nbsp;\n<a href="/charts/#dominance-percentage"><span class="js-global-stats-btc-dominance"></span>%</a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nCryptocurrencies:&nbsp;\n<a href="/all/views/all/"><span class="js-global-stats-cryptocurrencies"></span></a>\n</span>\n&nbsp;\xe2\x80\xa2&nbsp;\n<span>\nMarkets:&nbsp;\n<a href="/currencies/volume/24-hour/"><span class="js-global-stats-markets"></span></a>\n</span>\n</div>\n</div>\n</div>\n</div>\n</div>\n<nav class="cmc-navbar cmc-nav-desktop js-navbar">\n<div class="container">\n<div>\n<a href="/" title="Go to homepage">\n<span class="cmc-logo cmc-logo_color_black cmc-navbar__logo"></span>\n</a>\n</div>\n<ul class="cmc-navbar__menu" role="menu">\n<li class="cmc-navbar__menuitem js-nav-dropdown-open" data-menu-id="rankings" role="menuitem">\n<div>Rankings</div>\n<div class="cmc-nav-caret"><span class="icon-caret-down"></span></div>\n</li>\n<li class="cmc-navbar__menuitem js-nav-dropdown-open" data-menu-id="tools" role="menuitem">\n<div>Tools</div>\n<div class="cmc-nav-caret"><span class="icon-caret-down"></span></div>\n</li>\n<li class="cmc-navbar__menuitem js-nav-dropdown-open" data-menu-id="resources" role="menuitem">\n<div>Resources</div>\n<div class="cmc-nav-caret"><span class="icon-caret-down"></span></div>\n</li>\n<li class="cmc-navbar__menuitem" role="menuitem">\n<a href="https://blog.coinmarketcap.com" target="_blank" rel="noopener">Blog</a>\n</li>\n<li class="cmc-navbar__menuitem js-nav-dropdown-open" data-menu-id="more" role="menuitem">\n<div><span class="icon-ellipsis-h cmc-navbar-ellipsis"></span></div>\n<div class="cmc-nav-caret"><span class="icon-caret-down"></span></div>\n</li>\n</ul>\n<div class="cmc-quick-search js-nav-search">\n<form action="/search/" role="search">\n<div class="has-feedback">\n<input type="text" class="cmc-quick-search__input js-quick-search js-nav-search-input" placeholder="Search" name="q" aria-label="Search">\n<span class="icon-search cmc-quick-search__icon cmc-quick-search__icon--open js-nav-search-open" aria-hidden="true"></span>\n<span class="icon-times-circle cmc-quick-search__icon cmc-quick-search__icon--close is-inactive js-nav-search-close" aria-hidden="true"></span>\n</div>\n</form>\n</div>\n</div>\n</nav>\n<div class="clearfix"></div>\n<div class="cmc-nav-dropdown cmc-nav-dropdown-enter cmc-nav-desktop js-nav-dropdown">\n<div class="cmc-navsubmenu is-closed js-nav-dropdown-menu" data-menu-id="rankings">\n<div class="cmc-navsubmenu__content cmc-navsubmenu__content--size-sm">\n<div class="cmc-navsubmenu__section cmc-navsubmenu__section--layout-two-col-sm">\n<div class="w-100">\n<div class="cmc-navsubmenu__item">\n<h6 class="cmc-nav-heading">Market Cap</h6>\n<ul class="cmc-nav-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/">All Cryptocurrencies</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/coins/">Coins</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/tokens/">Tokens</a></li>\n</ul>\n</div>\n<div class="cmc-navsubmenu__item">\n<h6 class="cmc-nav-heading">Trends</h6>\n<ul class="cmc-nav-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/gainers-losers/">Gainers &amp; Losers</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/new/">Recently Added</a></li>\n</ul>\n</div>\n</div>\n<div class="w-100">\n<div class="cmc-navsubmenu__item">\n<h6 class="cmc-nav-heading">Cryptocurrencies</h6>\n<ul class="cmc-nav-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/currencies/volume/24-hour/">Daily Volume</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/currencies/volume/monthly/">Monthly Volume</a></li>\n</ul>\n</div>\n<div class="cmc-navsubmenu__item">\n<h6 class="cmc-nav-heading">Exchanges</h6>\n<ul class="cmc-nav-menu-list">\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/rankings/exchanges/">Adjusted Volume</a></li>\n<li><a class="cmc-nav-link cmc-nav-link--arrow" href="/rankings/exchanges/reported/">Reported Volume</a></li>\n</ul>\n</div>\n</div>\n</div>\n<div class="cmc-navsubmenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--according-to-cmc">\n<a class="cmc-nav-spotlight__link js-nav-track-click" href="https://blog.coinmarketcap.com/2018/11/27/according-to-coinmarketcap-2018-edition/?utm_source=coinmarketcap.com&utm_medium=web&utm_campaign=nav2" target="_blank" rel="noopener" title="According to CoinMarketCap blog post" data-linkid="According to CMC - desktop"></a>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-navsubmenu is-closed js-nav-dropdown-menu" data-menu-id="tools">\n<div class="cmc-navsubmenu__content">\n<div class="cmc-navsubmenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--tools">\n<a class="cmc-nav-spotlight__link" href="https://coinmarketcap.com/mobile/" target="_blank"></a>\n</div>\n</div>\n<div class="cmc-navsubmenu__section cmc-navsubmenu__section--layout-three-col">\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/charts/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Global Charts</h6>\n<p class="cmc-nav-desc">Total Market Cap &amp; Dominance Charts</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/converter/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Currency Converter</h6>\n<p class="cmc-nav-desc">Convert prices of thousands of crypto and fiat currencies</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/historical/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Historical Snapshots</h6>\n<p class="cmc-nav-desc">View crypto rankings from the past</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/widget/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Site Widgets</h6>\n<p class="cmc-nav-desc">Powerful &amp; reliable widgets for your site</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/watchlist/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Watchlist</h6>\n<p class="cmc-nav-desc">Keep an eye on your favorite cryptocurrencies</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="https://blockchain.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Blockchain Explorer</h6>\n<p class="cmc-nav-desc">Search and visualize blockchain data</p>\n</a>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-navsubmenu is-closed js-nav-dropdown-menu" data-menu-id="resources">\n<div class="cmc-navsubmenu__content">\n<div class="cmc-navsubmenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--resources">\n<a class="cmc-nav-spotlight__link" href="https://coinmarketcap.com/mobile/" target="_blank"></a>\n</div>\n</div>\n<div class="cmc-navsubmenu__section cmc-navsubmenu__section--layout-three-col">\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/api/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Crypto API</h6>\n<p class="cmc-nav-desc">A powerful &amp; flexible market data API for commercial &amp; personal use</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/#newsletter" target="_blank">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Newsletter</h6>\n<p class="cmc-nav-desc">Articles &amp; interviews for the crypto &amp; blockchain community</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/events/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Events Calendar</h6>\n<p class="cmc-nav-desc">A full list of crypto &amp; blockchain events from around the world</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/indices/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Crypto Indices</h6>\n<p class="cmc-nav-desc">The most comprehensive suite of institutional grade indices in the market</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/glossary/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Crypto Glossary</h6>\n<p class="cmc-nav-desc">Learn the terms, slang and definitions around the crypto space</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/intro-to-crypto/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Intro to Crypto</h6>\n<p class="cmc-nav-desc">Learn about the world of cryptocurrencies with our introductory guide</p>\n</a>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-navsubmenu is-closed js-nav-dropdown-menu" data-menu-id="more">\n<div class="cmc-navsubmenu__content">\n<div class="cmc-navsubmenu__section">\n<div class="cmc-nav-spotlight cmc-nav-spotlight--intro-to-crypto">\n<a class="cmc-nav-spotlight__link js-nav-track-click" href="/intro-to-crypto/" target="_blank" title="Intro to Crypto Guide" data-linkid="Intro to Crypto - desktop"></a>\n</div>\n</div>\n<div class="cmc-navsubmenu__section cmc-navsubmenu__section--layout-three-col">\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/methodology/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Methodology</h6>\n<p class="cmc-nav-desc">How CoinMarketCap analyzes data to offer up-to-the-minute updates</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/careers/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Careers</h6>\n<p class="cmc-nav-desc">Discover your next career! Build the future of crypto data with us</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="/faq/">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">FAQ</h6>\n<p class="cmc-nav-desc">Need help? Check our FAQ, with questions we are most commonly asked</p>\n</a>\n</div>\n<div class="cmc-navsubmenu__item">\n<a class="cmc-nav-link--menuitem" href="https://shop.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">\n<h6 class="cmc-nav-heading cmc-nav-link--menuitem__arrow">Shop</h6>\n<p class="cmc-nav-desc">Purchase exclusive CoinMarketCap apparel!</p>\n</a>\n</div>\n<hr>\n<div class="cmc-navsubmenu__item cmc-navsubmenu__item--two-col cmc-nav-footer-links">\n<div>\n<a class="cmc-link-secondary" href="/disclaimer/">Disclaimer</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary" href="/request/" target="_blank">Request Form</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary" href="/terms/">Terms of Use</a>\n</div>\n<div>\n<a class="cmc-link-secondary" href="/privacy/">Privacy Policy</a>\n&nbsp;\xe2\x80\xa2&nbsp;\n<a class="cmc-link-secondary js-nav-track-click" href="/advertising/" target="_blank" data-linkid="Advertise - more">Advertise</a>\n</div>\n</div>\n<div class="cmc-navsubmenu__item cmc-navsubmenu__item--two-col cmc-nav-social-links">\n<ul class="cmc-list-inline">\n<li><a href="https://twitter.com/CoinMarketCap" target="_blank" rel="noopener" title="Twitter"><span class="icon-twitter cmc-nav-social-icon"></span></a></li>\n<li><a href="https://www.facebook.com/CoinMarketCap" target="_blank" rel="noopener" title="Facebook"><span class="icon-facebook cmc-nav-social-icon"></span></a></li>\n<li><a href="https://t.me/CoinMarketCap" target="_blank" rel="nofollow noopener" title="Telegram"><span class="icon-telegram cmc-nav-social-icon"></span></a></li>\n<li><a href="https://www.linkedin.com/company/coinmarketcap/" target="_blank" rel="nofollow noopener" title="LinkedIn"><span class="icon-linkedin cmc-nav-social-icon"></span></a></li>\n<li><a href="https://www.instagram.com/coinmarketcap/" target="_blank" rel="nofollow noopener" title="Instagram"><span class="icon-instagram cmc-nav-social-icon"></span></a></li>\n<li><a href="https://www.reddit.com/r/CoinMarketCap" target="_blank" rel="nofollow noopener" title="Reddit"><span class="icon-reddit cmc-nav-social-icon"></span></a></li>\n</ul>\n</div>\n</div>\n</div>\n</div>\n<div class="cmc-nav-close js-nav-dropdown-close">\n<span class="cmc-nav-close-icon icon-times"></span>&nbsp;&thinsp;Close\n</div>\n</div>\n<div class="cmc-overlay is-closed js-overlay"></div>\n</div>\n</div>\n<div class="container main-section padding-top-1x">\n<div id="leaderboard" class="text-center">\n\n<div id="div-gpt-ad-1542211140769-0" class="responsive-leaderboard">\n<script>\n                                        googletag.cmd.push(function() { googletag.display(\'div-gpt-ad-1542211140769-0\'); });\n                                    </script>\n</div>\n</div>\n<div class="cmc-main-content">\n<div class="cmc-main-content__main">\n<div class="row">\n<div class="col-xs-12">\n<div class="header header-1x">\n<h1 class="text-center h2">\nTop 100 Cryptocurrencies by Market Capitalization\n</h1>\n</div>\n</div>\n</div>\n<div class="row bottom-margin-1x">\n<div class="col-xs-12">\n<div>\n<div class="pull-right margin-bottom--lv1">\n<div class="pull-left">\n<div id="currency-switch" class="btn-group">\n<button id="currency-switch-button" type="button" class="btn btn-default dropdown-toggle" data-toggle="dropdown">\nUSD <span class="caret"></span>\n</button>\n<ul class="dropdown-menu" role="menu">\n<li class="pointer price-toggle" data-fiat-item data-currency="usd" data-fiat>USD</li>\n<li class="pointer price-toggle" data-currency="usd">USD</li>\n<li class="pointer price-toggle" data-currency="btc" data-currencyid="bitcoin">BTC</li>\n<li class="pointer price-toggle" data-currency="eth" data-currencyid="ethereum">ETH</li>\n<li class="pointer price-toggle" data-currency="xrp" data-currencyid="ripple">XRP</li>\n<li class="pointer price-toggle" data-currency="bch" data-currencyid="bitcoin-cash">BCH</li>\n<li class="pointer price-toggle" data-currency="ltc" data-currencyid="litecoin">LTC</li>\n</ul>\n</div>\n</div>\n<ul class="pull-left margin-left--lv1 pagination top-paginator">\n<li><a href="/2/">Next 100 &rarr;</a></li>\n<li>\n<a href="/all/views/all/">View All</a>\n</li>\n</ul>\n</div>\n<div class="pull-left">\n<ul id="category-tabs" class="nav nav-tabs text-left no-border-bottom" role="tablist">\n<li class="visible-xs active">\n<a class="dropdown-toggle" data-toggle="dropdown" href="#">Rankings <span class="caret"></span></a>\n<ul class="dropdown-menu" role="menu">\n<li class="section-heading">All Cryptocurrencies</li>\n<li><a href="/">Top 100</a></li>\n<li><a href="/all/views/all/">Full List</a></li>\n<li class="section-heading">Coins Only</li>\n<li><a href="/coins/">Top 100</a></li>\n<li><a href="/coins/views/all/">Full List</a></li>\n<li class="divider"></li>\n<li><a href="/coins/">Market Cap by Circulating Supply</a></li>\n<li><a href="/coins/views/market-cap-by-total-supply/">Market Cap by Total Supply</a></li>\n<li><a href="/coins/views/filter-non-mineable/">Filter Non-Mineable</a></li>\n<li class="section-heading">Tokens Only</li>\n<li><a href="/tokens/">Top 100</a></li>\n<li><a href="/tokens/views/all/">Full List</a></li>\n<li class="divider"></li>\n<li><a href="/tokens/">Market Cap by Circulating Supply</a></li>\n<li><a href="/tokens/views/market-cap-by-total-supply/">Market Cap by Total Supply</a></li>\n<li class="section-heading visible-xs">Exchanges</li>\n<li class="visible-xs"><a href="/rankings/exchanges/">Top 100 By Adjusted Volume</a></li>\n<li class="visible-xs"><a href="/rankings/exchanges/reported/">Top 100 By Reported Volume</a></li>\n</ul>\n</li>\n<li class="hidden-xs active">\n<a class="dropdown-toggle" data-toggle="dropdown" href="#">Cryptocurrencies <span class="caret"></span></a>\n<ul class="dropdown-menu" role="menu">\n<li class="section-heading">All Cryptocurrencies</li>\n<li><a href="/">Top 100</a></li>\n<li><a href="/all/views/all/">Full List</a></li>\n<li class="section-heading">Coins Only</li>\n<li><a href="/coins/">Top 100</a></li>\n<li><a href="/coins/views/all/">Full List</a></li>\n<li class="divider"></li>\n<li><a href="/coins/">Market Cap by Circulating Supply</a></li>\n<li><a href="/coins/views/market-cap-by-total-supply/">Market Cap by Total Supply</a></li>\n<li><a href="/coins/views/filter-non-mineable/">Filter Non-Mineable</a></li>\n<li class="section-heading">Tokens Only</li>\n<li><a href="/tokens/">Top 100</a></li>\n<li><a href="/tokens/views/all/">Full List</a></li>\n<li class="divider"></li>\n<li><a href="/tokens/">Market Cap by Circulating Supply</a></li>\n<li><a href="/tokens/views/market-cap-by-total-supply/">Market Cap by Total Supply</a></li>\n</ul>\n</li>\n<li class="hidden-xs ">\n<a class="dropdown-toggle" data-toggle="dropdown" href="#">Exchanges <span class="caret"></span></a>\n<ul class="dropdown-menu" role="menu">\n<li><a href="/rankings/exchanges/">Top 100 By Adjusted Volume</a></li>\n<li><a href="/rankings/exchanges/reported/">Top 100 By Reported Volume</a></li>\n</ul>\n</li>\n<li role="presentation"><a href="/watchlist/">Watchlist</a></li>\n</ul>\n</div>\n</div>\n<div class="table-fixed-column-mobile compact-name-column">\n<table class="table floating-header " id="currencies">\n<thead>\n<tr>\n<th class="col-rank text-center sortable">#</th>\n<th id="th-name" class="sortable">Name</th>\n<th id="th-marketcap" class="sortable text-right" data-mobile-text="M. Cap">Market Cap</th>\n<th id="th-price" class="sortable text-right">Price</th>\n<th id="th-volume" class="sortable text-right" data-mobile-text="Volume">Volume (24h)</th>\n<th id="th-totalsupply" class="sortable text-right" title="The number of coins in existence available to the public" data-mobile-text="Supply">Circulating Supply</th>\n<th id="th-change" class="sortable text-right" data-mobile-text="Change">Change (24h)</th>\n<th id="th-marketcap-graph" class="text-right">Price Graph (7d)</th>\n<th id="th-more-options"></th>\n</tr>\n</thead>\n<tbody>\n<tr id="id-bitcoin" class="">\n<td class="text-center">\n1\n</td>\n<td class="no-wrap currency-name" data-sort="Bitcoin">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/1.png" class="logo-sprite" alt="Bitcoin" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitcoin/">BTC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitcoin/">Bitcoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1.87230971725e+11" data-btc="17924000.0" data-sort="1.87230971725e+11">\n$187,230,971,725\n</td>\n<td class="no-wrap text-right" data-sort="10445.8252469">\n<a href="/currencies/bitcoin/#markets" class="price" data-usd="10445.8252469" data-btc="1.0">$10445.83</a>\n</td>\n<td class="no-wrap text-right" data-sort="13629214823.4">\n<a href="/currencies/bitcoin/#markets" class="volume" data-usd="13629214823.4" data-btc="1303582.12308">$13,629,214,823</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17924000.0">\n<span data-supply="17924000.0">\n<span data-supply-container>17,924,000</span>\n<span class="hidden-xs">BTC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.752245" data-symbol="BTC" data-sort="-0.752245">-0.75%</td>\n<td><a href="/currencies/bitcoin/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1" data-cc-slug="bitcoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ethereum" class="">\n<td class="text-center">\n2\n</td>\n<td class="no-wrap currency-name" data-sort="Ethereum">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/1027.png" class="logo-sprite" alt="Ethereum" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ethereum/">ETH</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ethereum/">Ethereum</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="19568558583.0" data-btc="1871657.57337" data-sort="19568558583.0">\n$19,568,558,583\n</td>\n<td class="no-wrap text-right" data-sort="181.757853128">\n<a href="/currencies/ethereum/#markets" class="price" data-usd="181.757853128" data-btc="0.0173844415205">$181.76</a>\n</td>\n<td class="no-wrap text-right" data-sort="6465975081.48">\n<a href="/currencies/ethereum/#markets" class="volume" data-usd="6465975081.48" data-btc="618445.716333">$6,465,975,081</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="107662795.561">\n<span data-supply="107662795.561">\n<span data-supply-container>107,662,796</span>\n<span class="hidden-xs">ETH</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.65819" data-symbol="ETH" data-sort="1.65819">1.66%</td>\n<td><a href="/currencies/ethereum/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1027.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1027" data-cc-slug="ethereum">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1027" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1027">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ripple" class="">\n<td class="text-center">\n3\n</td>\n<td class="no-wrap currency-name" data-sort="XRP">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/52.png" class="logo-sprite" alt="XRP" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ripple/">XRP</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ripple/">XRP</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="11301041802.4" data-btc="1080928.8981" data-sort="11301041802.4">\n$11,301,041,802\n</td>\n<td class="no-wrap text-right" data-sort="0.262908740378">\n<a href="/currencies/ripple/#markets" class="price" data-usd="0.262908740378" data-btc="2.51468546003e-05">$0.262909</a>\n</td>\n<td class="no-wrap text-right" data-sort="1041705130.97">\n<a href="/currencies/ripple/#markets" class="volume" data-usd="1041705130.97" data-btc="99637.6439488">$1,041,705,131</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="42984656144.0">\n<span data-supply="42984656144.0">\n<span data-supply-container>42,984,656,144</span>\n<span class="hidden-xs">XRP</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.679351" data-symbol="XRP" data-sort="0.679351">0.68%</td>\n<td><a href="/currencies/ripple/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/52.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="52" data-cc-slug="ripple">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-52" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-52">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ripple/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ripple/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ripple/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bitcoin-cash" class="">\n<td class="text-center">\n4\n</td>\n<td class="no-wrap currency-name" data-sort="Bitcoin Cash">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/1831.png" class="logo-sprite" alt="Bitcoin Cash" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitcoin-cash/">BCH</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitcoin-cash/">Bitcoin Cash</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="5550641402.79" data-btc="530911.114227" data-sort="5550641402.79">\n$5,550,641,403\n</td>\n<td class="no-wrap text-right" data-sort="308.497293126">\n<a href="/currencies/bitcoin-cash/#markets" class="price" data-usd="308.497293126" data-btc="0.0295073361337">$308.50</a>\n</td>\n<td class="no-wrap text-right" data-sort="1365895040.64">\n<a href="/currencies/bitcoin-cash/#markets" class="volume" data-usd="1365895040.64" data-btc="130645.956985">$1,365,895,041</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17992512.5">\n<span data-supply="17992512.5">\n<span data-supply-container>17,992,513</span>\n<span class="hidden-xs">BCH</span>\n</span>\n</td>\n\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.84339" data-symbol="BCH" data-sort="1.84339">1.84%</td>\n<td><a href="/currencies/bitcoin-cash/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1831.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1831" data-cc-slug="bitcoin-cash">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1831" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1831">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-cash/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-cash/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-cash/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-litecoin" class="">\n<td class="text-center">\n5\n</td>\n<td class="no-wrap currency-name" data-sort="Litecoin">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/2.png" class="logo-sprite" alt="Litecoin" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/litecoin/">LTC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/litecoin/">Litecoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="4477350589.47" data-btc="428252.343783" data-sort="4477350589.47">\n$4,477,350,589\n</td>\n<td class="no-wrap text-right" data-sort="70.8302111677">\n<a href="/currencies/litecoin/#markets" class="price" data-usd="70.8302111677" data-btc="0.00677481098186">$70.83</a>\n</td>\n<td class="no-wrap text-right" data-sort="2783970810.21">\n<a href="/currencies/litecoin/#markets" class="volume" data-usd="2783970810.21" data-btc="266282.927966">$2,783,970,810</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="63212441.6355">\n<span data-supply="63212441.6355">\n <span data-supply-container>63,212,442</span>\n<span class="hidden-xs">LTC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.35123" data-symbol="LTC" data-sort="2.35123">2.35%</td>\n<td><a href="/currencies/litecoin/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2" data-cc-slug="litecoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/litecoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/litecoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/litecoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-tether" class="">\n<td class="text-center">\n6\n</td>\n<td class="no-wrap currency-name" data-sort="Tether">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/825.png" class="logo-sprite" alt="Tether" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/tether/">USDT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/tether/">Tether</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="4091812877.14" data-btc="391366.207574" data-sort="4091812877.14">\n$4,091,812,877\n</td>\n<td class="no-wrap text-right" data-sort="1.0050646841">\n<a href="/currencies/tether/#markets" class="price" data-usd="1.0050646841" data-btc="9.61305821144e-05">$1.01</a>\n</td>\n<td class="no-wrap text-right" data-sort="16229928830.2">\n<a href="/currencies/tether/#markets" class="volume" data-usd="16229928830.2" data-btc="1552330.44281">$16,229,928,830</a>\n</td>\n\n<td class="no-wrap text-right circulating-supply" data-sort="4071193567.81">\n<span data-supply="4071193567.81">\n<span data-supply-container>4,071,193,568</span>\n<span class="hidden-xs">USDT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.0254813" data-symbol="USDT" data-sort="0.0254813">0.03%</td>\n<td><a href="/currencies/tether/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/825.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="825" data-cc-slug="tether">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-825" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-825">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tether/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tether/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tether/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-binance-coin" class="">\n<td class="text-center">\n7\n</td>\n<td class="no-wrap currency-name" data-sort="Binance Coin">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/1839.png" class="logo-sprite" alt="Binance Coin" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/binance-coin/">BNB</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/binance-coin/">Binance Coin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="3512633214.83" data-btc="335978.471429" data-sort="3512633214.83">\n$3,512,633,215\n</td>\n<td class="no-wrap text-right" data-sort="22.5839491338">\n<a href="/currencies/binance-coin/#markets" class="price" data-usd="22.5839491338" data-btc="0.00216012325932">$22.58</a>\n </td>\n<td class="no-wrap text-right" data-sort="176785997.962">\n<a href="/currencies/binance-coin/#markets" class="volume" data-usd="176785997.962" data-btc="16909.3343178">$176,785,998</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="155536713.0">\n<span data-supply="155536713.0">\n<span data-supply-container>155,536,713</span>\n<span class="hidden-xs">BNB</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.0814117" data-symbol="BNB" data-sort="-0.0814117">-0.08%</td>\n<td><a href="/currencies/binance-coin/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1839.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1839" data-cc-slug="binance-coin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1839" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1839">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/binance-coin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/binance-coin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/binance-coin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-eos" class="">\n<td class="text-center">\n8\n</td>\n<td class="no-wrap currency-name" data-sort="EOS">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/1765.png" class="logo-sprite" alt="EOS" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/eos/">EOS</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/eos/">EOS</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="3511065747.94" data-btc="335828.545405" data-sort="3511065747.94">\n$3,511,065,748\n </td>\n<td class="no-wrap text-right" data-sort="3.77161905765">\n<a href="/currencies/eos/#markets" class="price" data-usd="3.77161905765" data-btc="0.000360750106345">$3.77</a>\n</td>\n<td class="no-wrap text-right" data-sort="2096720602.96">\n<a href="/currencies/eos/#markets" class="volume" data-usd="2096720602.96" data-btc="200548.403466">$2,096,720,603</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="930917384.357">\n<span data-supply="930917384.357">\n<span data-supply-container>930,917,384</span>\n<span class="hidden-xs">EOS</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="5.19103" data-symbol="EOS" data-sort="5.19103">5.19%</td>\n<td><a href="/currencies/eos/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1765.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1765" data-cc-slug="eos">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1765" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1765">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/eos/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/eos/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/eos/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bitcoin-sv" class="">\n<td class="text-center">\n9\n</td>\n<td class="no-wrap currency-name" data-sort="Bitcoin SV">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/3602.png" class="logo-sprite" alt="Bitcoin SV" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitcoin-sv/">BSV</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitcoin-sv/">Bitcoin SV</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="2420008477.87" data-btc="231470.438134" data-sort="2420008477.87">\n$2,420,008,478\n</td>\n<td class="no-wrap text-right" data-sort="135.536847334">\n<a href="/currencies/bitcoin-sv/#markets" class="price" data-usd="135.536847334" data-btc="0.0129639105493">$135.54</a>\n</td>\n<td class="no-wrap text-right" data-sort="306753584.466">\n<a href="/currencies/bitcoin-sv/#markets" class="volume" data-usd="306753584.466" data-btc="29340.5528306">$306,753,584</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17854985.7509">\n<span data-supply="17854985.7509">\n<span data-supply-container>17,854,986</span>\n<span class="hidden-xs">BSV</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.276185" data-symbol="BSV" data-sort="0.276185">0.28%</td>\n<td><a href="/currencies/bitcoin-sv/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3602.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3602" data-cc-slug="bitcoin-sv">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3602" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3602">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-sv/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-sv/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-sv/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-monero" class="">\n<td class="text-center">\n10\n</td>\n<td class="no-wrap currency-name" data-sort="Monero">\n<img src="https://s2.coinmarketcap.com/static/img/coins/32x32/328.png" class="logo-sprite" alt="Monero" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/monero/">XMR</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/monero/">Monero</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1332090844.01" data-btc="127412.632689" data-sort="1332090844.01">\n$1,332,090,844\n</td>\n<td class="no-wrap text-right" data-sort="77.4544506162">\n<a href="/currencies/monero/#markets" class="price" data-usd="77.4544506162" data-btc="0.00740841025289">$77.45</a>\n</td>\n<td class="no-wrap text-right" data-sort="58301291.312">\n<a href="/currencies/monero/#markets" class="volume" data-usd="58301291.312" data-btc="5576.43725927">$58,301,291</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17198377.0256">\n<span data-supply="17198377.0256">\n<span data-supply-container>17,198,377</span>\n<span class="hidden-xs">XMR</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-1.28874" data-symbol="XMR" data-sort="-1.28874">-1.29%</td>\n<td><a href="/currencies/monero/#charts">\n<img class="sparkline" alt="sparkline" src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/328.png">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="328" data-cc-slug="monero">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-328" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-328">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monero/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monero/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monero/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-cardano" class="">\n<td class="text-center">\n11\n</td>\n<td class="no-wrap currency-name" data-sort="Cardano">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2010.png" class="logo-sprite lazyload" alt="Cardano" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/cardano/">ADA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/cardano/">Cardano</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1212509110.8" data-btc="115974.806569" data-sort="1212509110.8">\n$1,212,509,111\n</td>\n<td class="no-wrap text-right" data-sort="0.0467661438657">\n<a href="/currencies/cardano/#markets" class="price" data-usd="0.0467661438657" data-btc="4.47311648261e-06">$0.046766</a>\n</td>\n<td class="no-wrap text-right" data-sort="46428263.5825">\n<a href="/currencies/cardano/#markets" class="volume" data-usd="46428263.5825" data-btc="4440.79870442">$46,428,264</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="25927070538.0">\n<span data-supply="25927070538.0">\n<span data-supply-container>25,927,070,538</span>\n<span class="hidden-xs">ADA</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.06702" data-symbol="ADA" data-sort="1.06702">1.07%</td>\n<td><a href="/currencies/cardano/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2010.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2010" data-cc-slug="cardano">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2010" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2010">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cardano/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cardano/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cardano/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-stellar" class="">\n<td class="text-center">\n12\n</td>\n<td class="no-wrap currency-name" data-sort="Stellar">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/512.png" class="logo-sprite lazyload" alt="Stellar" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/stellar/">XLM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/stellar/">Stellar</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1180805533.75" data-btc="112942.40361" data-sort="1180805533.75">\n$1,180,805,534\n</td>\n<td class="no-wrap text-right" data-sort="0.0597954600065">\n<a href="/currencies/stellar/#markets" class="price" data-usd="0.0597954600065" data-btc="5.71935241247e-06">$0.059795</a>\n</td>\n<td class="no-wrap text-right" data-sort="161647178.087">\n<a href="/currencies/stellar/#markets" class="volume" data-usd="161647178.087" data-btc="15461.3272958">$161,647,178</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="19747411151.6">\n<span data-supply="19747411151.6">\n<span data-supply-container>19,747,411,152</span>\n<span class="hidden-xs">XLM</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.491586" data-symbol="XLM" data-sort="0.491586">0.49%</td>\n<td><a href="/currencies/stellar/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/512.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="512" data-cc-slug="stellar">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-512" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-512">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stellar/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stellar/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stellar/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-unus-sed-leo" class="">\n<td class="text-center">\n13\n</td>\n<td class="no-wrap currency-name" data-sort="UNUS SED LEO">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3957.png" class="logo-sprite lazyload" alt="UNUS SED LEO" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/unus-sed-leo/">LEO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/unus-sed-leo/">UNUS SED LEO</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1069184893.85" data-btc="102266.04497" data-sort="1069184893.85">\n$1,069,184,894\n</td>\n<td class="no-wrap text-right" data-sort="1.06972093861">\n<a href="/currencies/unus-sed-leo/#markets" class="price" data-usd="1.06972093861" data-btc="0.000102317316904">$1.07</a>\n</td>\n<td class="no-wrap text-right" data-sort="5236609.78173">\n<a href="/currencies/unus-sed-leo/#markets" class="volume" data-usd="5236609.78173" data-btc="500.87442734">$5,236,610</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="999498892.9">\n<span data-supply="999498892.9">\n<span data-supply-container>999,498,893</span>\n<span class="hidden-xs">LEO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.495265" data-symbol="LEO" data-sort="0.495265">0.50%</td>\n<td><a href="/currencies/unus-sed-leo/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3957.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3957" data-cc-slug="unus-sed-leo">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3957" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3957">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/unus-sed-leo/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/unus-sed-leo/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/unus-sed-leo/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-tron" class="">\n<td class="text-center">\n14\n</td>\n<td class="no-wrap currency-name" data-sort="TRON">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1958.png" class="logo-sprite lazyload" alt="TRON" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/tron/">TRX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/tron/">TRON</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="1056225130.62" data-btc="101026.461679" data-sort="1056225130.62">\n$1,056,225,131\n</td>\n<td class="no-wrap text-right" data-sort="0.0158397166721">\n<a href="/currencies/tron/#markets" class="price" data-usd="0.0158397166721" data-btc="1.51504682381e-06">$0.015840</a>\n</td>\n<td class="no-wrap text-right" data-sort="564691577.633">\n<a href="/currencies/tron/#markets" class="volume" data-usd="564691577.633" data-btc="54011.9623879">$564,691,578</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="66682072191.4">\n<span data-supply="66682072191.4">\n<span data-supply-container>66,682,072,191</span>\n<span class="hidden-xs">TRX</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.14691" data-symbol="TRX" data-sort="2.14691">2.15%</td>\n<td><a href="/currencies/tron/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1958.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1958" data-cc-slug="tron">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1958" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1958">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tron/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tron/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tron/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-huobi-token" class="">\n<td class="text-center">\n15\n</td>\n<td class="no-wrap currency-name" data-sort="Huobi Token">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2502.png" class="logo-sprite lazyload" alt="Huobi Token" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/huobi-token/">HT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/huobi-token/">Huobi Token</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="998567383.336" data-btc="95511.5785093" data-sort="998567383.336">\n$998,567,383\n</td>\n<td class="no-wrap text-right" data-sort="4.06118856281">\n<a href="/currencies/huobi-token/#markets" class="price" data-usd="4.06118856281" data-btc="0.00038844702594">$4.06</a>\n</td>\n<td class="no-wrap text-right" data-sort="68776975.5773">\n<a href="/currencies/huobi-token/#markets" class="volume" data-usd="68776975.5773" data-btc="6578.42185925">$68,776,976</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="245880576.066">\n<span data-supply="245880576.066">\n<span data-supply-container>245,880,576</span>\n<span class="hidden-xs">HT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.587206" data-symbol="HT" data-sort="0.587206">0.59%</td>\n<td><a href="/currencies/huobi-token/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2502.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2502" data-cc-slug="huobi-token">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2502" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2502">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/huobi-token/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/huobi-token/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/huobi-token/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-dash" class="">\n<td class="text-center">\n16\n</td>\n<td class="no-wrap currency-name" data-sort="Dash">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/131.png" class="logo-sprite lazyload" alt="Dash" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/dash/">DASH</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/dash/">Dash</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="793031755.041" data-btc="75852.3821185" data-sort="793031755.041">\n$793,031,755\n</td>\n<td class="no-wrap text-right" data-sort="87.8354902244">\n<a href="/currencies/dash/#markets" class="price" data-usd="87.8354902244" data-btc="0.00840134222333">$87.84</a>\n</td>\n<td class="no-wrap text-right" data-sort="170461263.19">\n<a href="/currencies/dash/#markets" class="volume" data-usd="170461263.19" data-btc="16304.3822516">$170,461,263</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="9028602.82347">\n <span data-supply="9028602.82347">\n<span data-supply-container>9,028,603</span>\n<span class="hidden-xs">DASH</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.88528" data-symbol="DASH" data-sort="3.88528">3.89%</td>\n<td><a href="/currencies/dash/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/131.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="131" data-cc-slug="dash">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-131" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-131">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dash/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dash/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dash/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ethereum-classic" class="">\n<td class="text-center">\n17\n</td>\n<td class="no-wrap currency-name" data-sort="Ethereum Classic">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1321.png" class="logo-sprite lazyload" alt="Ethereum Classic" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ethereum-classic/">ETC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ethereum-classic/">Ethereum Classic</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="755620187.876" data-btc="72274.0178598" data-sort="755620187.876">\n$755,620,188\n</td>\n<td class="no-wrap text-right" data-sort="6.66794098836">\n<a href="/currencies/ethereum-classic/#markets" class="price" data-usd="6.66794098836" data-btc="0.000637779262403">$6.67</a>\n</td>\n<td class="no-wrap text-right" data-sort="510984205.821">\n<a href="/currencies/ethereum-classic/#markets" class="volume" data-usd="510984205.821" data-btc="48874.9271263">$510,984,206</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="113321367.0">\n<span data-supply="113321367.0">\n<span data-supply-container>113,321,367</span>\n<span class="hidden-xs">ETC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.453472" data-symbol="ETC" data-sort="-0.453472">-0.45%</td>\n<td><a href="/currencies/ethereum-classic/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1321.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1321" data-cc-slug="ethereum-classic">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1321" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1321">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum-classic/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum-classic/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ethereum-classic/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-tezos" class="">\n<td class="text-center">\n18\n</td>\n<td class="no-wrap currency-name" data-sort="Tezos">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2011.png" class="logo-sprite lazyload" alt="Tezos" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/tezos/">XTZ</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/tezos/">Tezos</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="702942628.097" data-btc="67235.4826844" data-sort="702942628.097">\n$702,942,628\n</td>\n<td class="no-wrap text-right" data-sort="1.0644620187">\n<a href="/currencies/tezos/#markets" class="price" data-usd="1.0644620187" data-btc="0.000101814308545">$1.06</a>\n</td>\n<td class="no-wrap text-right" data-sort="10245034.2376">\n<a href="/currencies/tezos/#markets" class="volume" data-usd="10245034.2376" data-btc="979.923246289">$10,245,034</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="660373611.973">\n<span data-supply="660373611.973">\n<span data-supply-container>660,373,612</span>\n<span class="hidden-xs">XTZ</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.49813" data-symbol="XTZ" data-sort="3.49813">3.50%</td>\n<td><a href="/currencies/tezos/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2011.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2011" data-cc-slug="tezos">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2011" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2011">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tezos/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tezos/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/tezos/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-iota" class="">\n<td class="text-center">\n19\n</td>\n<td class="no-wrap currency-name" data-sort="IOTA">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1720.png" class="logo-sprite lazyload" alt="IOTA" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/iota/">MIOTA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/iota/">IOTA</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="680830610.023" data-btc="65120.4989726" data-sort="680830610.023">\n$680,830,610\n</td>\n<td class="no-wrap text-right" data-sort="0.244944483673">\n<a href="/currencies/iota/#markets" class="price" data-usd="0.244944483673" data-btc="2.34285984833e-05">$0.244944</a>\n</td>\n<td class="no-wrap text-right" data-sort="3978662.72307">\n<a href="/currencies/iota/#markets" class="volume" data-usd="3978662.72307" data-btc="380.553544385">$3,978,663</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="2779530283.0">\n<span data-supply="2779530283.0">\n<span data-supply-container>2,779,530,283</span>\n<span class="hidden-xs">MIOTA</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.10662" data-symbol="MIOTA" data-sort="1.10662">1.11%</td>\n<td><a href="/currencies/iota/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1720.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1720" data-cc-slug="iota">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1720" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1720">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iota/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iota/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iota/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-neo" class="">\n<td class="text-center">\n20\n</td>\n<td class="no-wrap currency-name" data-sort="NEO">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1376.png" class="logo-sprite lazyload" alt="NEO" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/neo/">NEO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/neo/">NEO</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="657714002.684" data-btc="62909.4276983" data-sort="657714002.684">\n$657,714,003\n</td>\n<td class="no-wrap text-right" data-sort="9.32414094988">\n<a href="/currencies/neo/#markets" class="price" data-usd="9.32414094988" data-btc="0.000891841086767">$9.32</a>\n</td>\n<td class="no-wrap text-right" data-sort="242275387.306">\n<a href="/currencies/neo/#markets" class="volume" data-usd="242275387.306" data-btc="23173.3031358">$242,275,387</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="70538831.0">\n<span data-supply="70538831.0">\n<span data-supply-container>70,538,831</span>\n<span class="hidden-xs">NEO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.05839" data-symbol="NEO" data-sort="2.05839">2.06%</td>\n<td><a href="/currencies/neo/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1376.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1376" data-cc-slug="neo">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1376" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1376">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/neo/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/neo/#markets">View Markets</a></li>\n <li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/neo/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-chainlink" class="">\n<td class="text-center">\n21\n</td>\n<td class="no-wrap currency-name" data-sort="Chainlink">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1975.png" class="logo-sprite lazyload" alt="Chainlink" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/chainlink/">LINK</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/chainlink/">Chainlink</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="644019550.122" data-btc="61599.5723966" data-sort="644019550.122">\n$644,019,550\n</td>\n<td class="no-wrap text-right" data-sort="1.84005585749">\n<a href="/currencies/chainlink/#markets" class="price" data-usd="1.84005585749" data-btc="0.000175998778276">$1.84</a>\n</td>\n<td class="no-wrap text-right" data-sort="88600248.4847">\n<a href="/currencies/chainlink/#markets" class="volume" data-usd="88600248.4847" data-btc="8474.49028507">$88,600,248</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="350000000.0">\n<span data-supply="350000000.0">\n<span data-supply-container>350,000,000</span>\n<span class="hidden-xs">LINK</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.34382" data-symbol="LINK" data-sort="3.34382">3.34%</td>\n<td><a href="/currencies/chainlink/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1975.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1975" data-cc-slug="chainlink">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1975" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1975">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/chainlink/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/chainlink/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/chainlink/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-cosmos" class="">\n<td class="text-center">\n22\n</td>\n<td class="no-wrap currency-name" data-sort="Cosmos">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3794.png" class="logo-sprite lazyload" alt="Cosmos" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/cosmos/">ATOM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/cosmos/">Cosmos</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="521258265.186" data-btc="49857.6265856" data-sort="521258265.186">\n$521,258,265\n</td>\n<td class="no-wrap text-right" data-sort="2.73355987061">\n<a href="/currencies/cosmos/#markets" class="price" data-usd="2.73355987061" data-btc="0.000261461191852">$2.73</a>\n</td>\n<td class="no-wrap text-right" data-sort="195841911.743">\n<a href="/currencies/cosmos/#markets" class="volume" data-usd="195841911.743" data-btc="18732.0059126">$195,841,912</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="190688439.2">\n<span data-supply="190688439.2">\n<span data-supply-container>190,688,439</span>\n<span class="hidden-xs">ATOM</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="24.299" data-symbol="ATOM" data-sort="24.299">24.30%</td>\n<td><a href="/currencies/cosmos/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3794.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3794" data-cc-slug="cosmos">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3794" data-toggle="dropdown">\n <span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3794">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cosmos/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cosmos/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/cosmos/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-maker" class="">\n<td class="text-center">\n23\n</td>\n<td class="no-wrap currency-name" data-sort="Maker">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1518.png" class="logo-sprite lazyload" alt="Maker" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/maker/">MKR</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/maker/">Maker</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="448506566.225" data-btc="42899.0279742" data-sort="448506566.225">\n$448,506,566\n</td>\n<td class="no-wrap text-right" data-sort="448.506566225">\n<a href="/currencies/maker/#markets" class="price" data-usd="448.506566225" data-btc="0.0428990279742">$448.51</a>\n</td>\n<td class="no-wrap text-right" data-sort="12845481.653">\n<a href="/currencies/maker/#markets" class="volume" data-usd="12845481.653" data-btc="1228.65241732">$12,845,482</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1000000.0">\n<span data-supply="1000000.0">\n<span data-supply-container>1,000,000</span>\n<span class="hidden-xs">MKR</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.32908" data-symbol="MKR" data-sort="2.32908">2.33%</td>\n<td><a href="/currencies/maker/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1518.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1518" data-cc-slug="maker">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1518" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1518">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maker/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maker/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maker/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-usd-coin" class="">\n<td class="text-center">\n24\n</td>\n<td class="no-wrap currency-name" data-sort="USD Coin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3408.png" class="logo-sprite lazyload" alt="USD Coin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/usd-coin/">USDC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/usd-coin/">USD Coin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="443613230.429" data-btc="42430.9872252" data-sort="443613230.429">\n$443,613,230\n</td>\n<td class="no-wrap text-right" data-sort="1.00279535207">\n<a href="/currencies/usd-coin/#markets" class="price" data-usd="1.00279535207" data-btc="9.59159778262e-05">$1.00</a>\n</td>\n<td class="no-wrap text-right" data-sort="183066126.671">\n<a href="/currencies/usd-coin/#markets" class="volume" data-usd="183066126.671" data-btc="17510.0198762">$183,066,127</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="442376631.995">\n<span data-supply="442376631.995">\n<span data-supply-container>442,376,632</span>\n<span class="hidden-xs">USDC</span>\n</span>\n*\n </td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.144941" data-symbol="USDC" data-sort="0.144941">0.14%</td>\n<td><a href="/currencies/usd-coin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3408.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3408" data-cc-slug="usd-coin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3408" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3408">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/usd-coin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/usd-coin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/usd-coin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-nem" class="">\n<td class="text-center">\n25\n</td>\n<td class="no-wrap currency-name" data-sort="NEM">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/873.png" class="logo-sprite lazyload" alt="NEM" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/nem/">XEM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/nem/">NEM</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="424022836.496" data-btc="40557.1933487" data-sort="424022836.496">\n$424,022,836\n</td>\n<td class="no-wrap text-right" data-sort="0.0471136485048">\n<a href="/currencies/nem/#markets" class="price" data-usd="0.0471136485048" data-btc="4.50635481702e-06">$0.047114</a>\n</td>\n<td class="no-wrap text-right" data-sort="14640826.1422">\n<a href="/currencies/nem/#markets" class="volume" data-usd="14640826.1422" data-btc="1400.37461553">$14,640,826</a>\n</td>\n <td class="no-wrap text-right circulating-supply" data-sort="8999999999.0">\n<span data-supply="8999999999.0">\n<span data-supply-container>8,999,999,999</span>\n<span class="hidden-xs">XEM</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.330744" data-symbol="XEM" data-sort="0.330744">0.33%</td>\n<td><a href="/currencies/nem/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/873.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="873" data-cc-slug="nem">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-873" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-873">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nem/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nem/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nem/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ontology" class="">\n<td class="text-center">\n26\n</td>\n<td class="no-wrap currency-name" data-sort="Ontology">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2566.png" class="logo-sprite lazyload" alt="Ontology" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ontology/">ONT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ontology/">Ontology</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="405590353.77" data-btc="38794.1520653" data-sort="405590353.77">\n$405,590,354\n</td>\n<td class="no-wrap text-right" data-sort="0.760409721445">\n<a href="/currencies/ontology/#markets" class="price" data-usd="0.760409721445" data-btc="7.27321300704e-05">$0.760410</a>\n</td>\n<td class="no-wrap text-right" data-sort="80817255.4116">\n<a href="/currencies/ontology/#markets" class="volume" data-usd="80817255.4116" data-btc="7730.05784481">$80,817,255</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="533383967.0">\n<span data-supply="533383967.0">\n<span data-supply-container>533,383,967</span>\n<span class="hidden-xs">ONT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="6.01712" data-symbol="ONT" data-sort="6.01712">6.02%</td>\n<td><a href="/currencies/ontology/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2566.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2566" data-cc-slug="ontology">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2566" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2566">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ontology/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ontology/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ontology/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-crypto-com-chain" class="">\n<td class="text-center">\n27\n</td>\n<td class="no-wrap currency-name" data-sort="Crypto.com Chain">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3635.png" class="logo-sprite lazyload" alt="Crypto.com Chain" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/crypto-com-chain/">CRO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/crypto-com-chain/">Crypto.com Chain</a>\n</td>\n\n<td class="no-wrap market-cap text-right" data-usd="381468608.28" data-btc="36486.9407278" data-sort="381468608.28">\n$381,468,608\n</td>\n<td class="no-wrap text-right" data-sort="0.0394715923521">\n<a href="/currencies/crypto-com-chain/#markets" class="price" data-usd="0.0394715923521" data-btc="3.77540279678e-06">$0.039472</a>\n</td>\n<td class="no-wrap text-right" data-sort="9912725.23041">\n<a href="/currencies/crypto-com-chain/#markets" class="volume" data-usd="9912725.23041" data-btc="948.138352888">$9,912,725</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="9664383561.64">\n<span data-supply="9664383561.64">\n<span data-supply-container>9,664,383,562</span>\n<span class="hidden-xs">CRO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-4.38783" data-symbol="CRO" data-sort="-4.38783">-4.39%</td>\n<td><a href="/currencies/crypto-com-chain/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3635.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3635" data-cc-slug="crypto-com-chain">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3635" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3635">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com-chain/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com-chain/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com-chain/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-zcash" class="">\n<td class="text-center">\n28\n</td>\n<td class="no-wrap currency-name" data-sort="Zcash">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1437.png" class="logo-sprite lazyload" alt="Zcash" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/zcash/">ZEC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/zcash/">Zcash</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="357609759.597" data-btc="34204.8751035" data-sort="357609759.597">\n$357,609,760\n</td>\n<td class="no-wrap text-right" data-sort="48.5103019141">\n<a href="/currencies/zcash/#markets" class="price" data-usd="48.5103019141" data-btc="0.0046399427691">$48.51</a>\n</td>\n<td class="no-wrap text-right" data-sort="136127896.461">\n<a href="/currencies/zcash/#markets" class="volume" data-usd="136127896.461" data-btc="13020.4435745">$136,127,896</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="7371831.25">\n<span data-supply="7371831.25">\n<span data-supply-container>7,371,831</span>\n<span class="hidden-xs">ZEC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.80891" data-symbol="ZEC" data-sort="2.80891">2.81%</td>\n<td><a href="/currencies/zcash/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1437.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1437" data-cc-slug="zcash">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1437" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1437">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcash/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcash/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcash/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-dogecoin" class="">\n<td class="text-center">\n29\n</td>\n<td class="no-wrap currency-name" data-sort="Dogecoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/74.png" class="logo-sprite lazyload" alt="Dogecoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/dogecoin/">DOGE</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/dogecoin/">Dogecoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="303857599.98" data-btc="29063.5559506" data-sort="303857599.98">\n$303,857,600\n</td>\n<td class="no-wrap text-right" data-sort="0.00250872458071">\n<a href="/currencies/dogecoin/#markets" class="price" data-usd="0.00250872458071" data-btc="2.39956009726e-07">$0.002509</a>\n</td>\n<td class="no-wrap text-right" data-sort="31595097.1732">\n<a href="/currencies/dogecoin/#markets" class="volume" data-usd="31595097.1732" data-btc="3022.02701041">$31,595,097</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1.21120350283e+11">\n<span data-supply="1.21120350283e+11">\n<span data-supply-container>121,120,350,283</span>\n<span class="hidden-xs">DOGE</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.48771" data-symbol="DOGE" data-sort="-0.48771">-0.49%</td>\n<td><a href="/currencies/dogecoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/74.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="74" data-cc-slug="dogecoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-74" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-74">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dogecoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dogecoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dogecoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-v-systems" class="">\n<td class="text-center">\n30\n</td>\n<td class="no-wrap currency-name" data-sort="V Systems">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3704.png" class="logo-sprite lazyload" alt="V Systems" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/v-systems/">VSYS</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/v-systems/">V Systems</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="274208570.034" data-btc="26227.6675583" data-sort="274208570.034">\n$274,208,570\n</td>\n<td class="no-wrap text-right" data-sort="0.152375000535">\n<a href="/currencies/v-systems/#markets" class="price" data-usd="0.152375000535" data-btc="1.45744564356e-05">$0.152375</a>\n</td>\n<td class="no-wrap text-right" data-sort="6410616.88128">\n<a href="/currencies/v-systems/#markets" class="volume" data-usd="6410616.88128" data-btc="613.166570194">$6,410,617</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1799564030.0">\n<span data-supply="1799564030.0">\n<span data-supply-container>1,799,564,030</span>\n<span class="hidden-xs">VSYS</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.751162" data-symbol="VSYS" data-sort="0.751162">0.75%</td>\n<td><a href="/currencies/v-systems/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3704.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3704" data-cc-slug="v-systems">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3704" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3704">\n <li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/v-systems/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/v-systems/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/v-systems/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-hedgetrade" class="">\n<td class="text-center">\n31\n</td>\n<td class="no-wrap currency-name" data-sort="HedgeTrade">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3662.png" class="logo-sprite lazyload" alt="HedgeTrade" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/hedgetrade/">HEDG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/hedgetrade/">HedgeTrade</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="270642723.405" data-btc="25886.599298" data-sort="270642723.405">\n$270,642,723\n</td>\n<td class="no-wrap text-right" data-sort="0.938449928403">\n<a href="/currencies/hedgetrade/#markets" class="price" data-usd="0.938449928403" data-btc="8.97614277306e-05">$0.938450</a>\n</td>\n<td class="no-wrap text-right" data-sort="560394.938912">\n<a href="/currencies/hedgetrade/#markets" class="volume" data-usd="560394.938912" data-btc="53.6009948824">$560,395</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="288393355.057">\n<span data-supply="288393355.057">\n<span data-supply-container>288,393,355</span>\n<span class="hidden-xs">HEDG</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.734798" data-symbol="HEDG" data-sort="0.734798">0.73%</td>\n<td><a href="/currencies/hedgetrade/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3662.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n </a></td>\n<td class="dropdown" data-more-options data-cc-id="3662" data-cc-slug="hedgetrade">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3662" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3662">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hedgetrade/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hedgetrade/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hedgetrade/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-decred" class="">\n<td class="text-center">\n32\n</td>\n<td class="no-wrap currency-name" data-sort="Decred">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1168.png" class="logo-sprite lazyload" alt="Decred" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/decred/">DCR</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/decred/">Decred</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="259220963.425" data-btc="24794.1238744" data-sort="259220963.425">\n$259,220,963\n</td>\n<td class="no-wrap text-right" data-sort="25.0285720229">\n<a href="/currencies/decred/#markets" class="price" data-usd="25.0285720229" data-btc="0.00239394803158">$25.03</a>\n</td>\n<td class="no-wrap text-right" data-sort="9197406.21393">\n<a href="/currencies/decred/#markets" class="volume" data-usd="9197406.21393" data-btc="879.719085904">$9,197,406</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="10357001.7174">\n<span data-supply="10357001.7174">\n<span data-supply-container>10,357,002</span>\n<span class="hidden-xs">DCR</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.14629" data-symbol="DCR" data-sort="2.14629">2.15%</td>\n<td><a href="/currencies/decred/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1168.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1168" data-cc-slug="decred">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1168" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1168">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/decred/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/decred/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/decred/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-paxos-standard-token" class="">\n<td class="text-center">\n33\n</td>\n<td class="no-wrap currency-name" data-sort="Paxos Standard Token">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3330.png" class="logo-sprite lazyload" alt="Paxos Standard Token" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/paxos-standard-token/">PAX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/paxos-standard-token/">Paxos Standar...</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="238415500.367" data-btc="22804.1103295" data-sort="238415500.367">\n$238,415,500\n</td>\n<td class="no-wrap text-right" data-sort="1.00361370759">\n<a href="/currencies/paxos-standard-token/#markets" class="price" data-usd="1.00361370759" data-btc="9.59942523906e-05">$1.00</a>\n</td>\n<td class="no-wrap text-right" data-sort="335041298.594">\n<a href="/currencies/paxos-standard-token/#markets" class="volume" data-usd="335041298.594" data-btc="32046.2332622">$335,041,299</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="237557038.694">\n<span data-supply="237557038.694">\n<span data-supply-container>237,557,039</span>\n<span class="hidden-xs">PAX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.198677" data-symbol="PAX" data-sort="0.198677">0.20%</td>\n<td><a href="/currencies/paxos-standard-token/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3330.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3330" data-cc-slug="paxos-standard-token">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3330" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3330">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/paxos-standard-token/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/paxos-standard-token/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/paxos-standard-token/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-basic-attention-token" class="">\n<td class="text-center">\n34\n</td>\n<td class="no-wrap currency-name" data-sort="Basic Attention Token">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1697.png" class="logo-sprite lazyload" alt="Basic Attention Token" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/basic-attention-token/">BAT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/basic-attention-token/">Basic Attenti...</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="234376428.486" data-btc="22417.7787335" data-sort="234376428.486">\n$234,376,428\n</td>\n <td class="no-wrap text-right" data-sort="0.176259253922">\n<a href="/currencies/basic-attention-token/#markets" class="price" data-usd="0.176259253922" data-btc="1.68589519845e-05">$0.176259</a>\n</td>\n<td class="no-wrap text-right" data-sort="20627375.6836">\n<a href="/currencies/basic-attention-token/#markets" class="volume" data-usd="20627375.6836" data-btc="1972.97973569">$20,627,376</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1329725522.33">\n<span data-supply="1329725522.33">\n<span data-supply-container>1,329,725,522</span>\n<span class="hidden-xs">BAT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.928629" data-symbol="BAT" data-sort="0.928629">0.93%</td>\n<td><a href="/currencies/basic-attention-token/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1697.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1697" data-cc-slug="basic-attention-token">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1697" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1697">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/basic-attention-token/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/basic-attention-token/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/basic-attention-token/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-vechain" class="">\n<td class="text-center">\n35\n</td>\n<td class="no-wrap currency-name" data-sort="VeChain">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3077.png" class="logo-sprite lazyload" alt="VeChain" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/vechain/">VET</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/vechain/">VeChain</a>\n </td>\n<td class="no-wrap market-cap text-right" data-usd="222399720.354" data-btc="21272.2232926" data-sort="222399720.354">\n$222,399,720\n</td>\n<td class="no-wrap text-right" data-sort="0.00401047306702">\n<a href="/currencies/vechain/#markets" class="price" data-usd="0.00401047306702" data-btc="3.8359615945e-07">$0.004010</a>\n</td>\n<td class="no-wrap text-right" data-sort="28420724.9393">\n<a href="/currencies/vechain/#markets" class="volume" data-usd="28420724.9393" data-btc="2718.40273038">$28,420,725</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="55454734800.0">\n<span data-supply="55454734800.0">\n<span data-supply-container>55,454,734,800</span>\n<span class="hidden-xs">VET</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.5964" data-symbol="VET" data-sort="3.5964">3.60%</td>\n<td><a href="/currencies/vechain/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3077.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3077" data-cc-slug="vechain">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3077" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3077">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/vechain/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/vechain/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/vechain/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-qtum" class="">\n<td class="text-center">\n36\n</td>\n<td class="no-wrap currency-name" data-sort="Qtum">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1684.png" class="logo-sprite lazyload" alt="Qtum" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/qtum/">QTUM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/qtum/">Qtum</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="203804472.375" data-btc="19493.6137396" data-sort="203804472.375">\n$203,804,472\n</td>\n<td class="no-wrap text-right" data-sort="2.12289381652">\n<a href="/currencies/qtum/#markets" class="price" data-usd="2.12289381652" data-btc="0.000203051834865">$2.12</a>\n</td>\n<td class="no-wrap text-right" data-sort="171982717.093">\n<a href="/currencies/qtum/#markets" class="volume" data-usd="171982717.093" data-btc="16449.907197">$171,982,717</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="96003140.0485">\n<span data-supply="96003140.0485">\n<span data-supply-container>96,003,140</span>\n<span class="hidden-xs">QTUM</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="4.37429" data-symbol="QTUM" data-sort="4.37429">4.37%</td>\n<td><a href="/currencies/qtum/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1684.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1684" data-cc-slug="qtum">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1684" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1684">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/qtum/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/qtum/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/qtum/historical-data/">View Historical Data</a></li>\n </ul>\n</td>\n</tr>\n<tr id="id-trueusd" class="">\n<td class="text-center">\n37\n</td>\n<td class="no-wrap currency-name" data-sort="TrueUSD">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2563.png" class="logo-sprite lazyload" alt="TrueUSD" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/trueusd/">TUSD</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/trueusd/">TrueUSD</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="199640836.679" data-btc="19095.3677882" data-sort="199640836.679">\n$199,640,837\n</td>\n<td class="no-wrap text-right" data-sort="1.00388358869">\n<a href="/currencies/trueusd/#markets" class="price" data-usd="1.00388358869" data-btc="9.60200661424e-05">$1.00</a>\n</td>\n<td class="no-wrap text-right" data-sort="298776655.212">\n<a href="/currencies/trueusd/#markets" class="volume" data-usd="298776655.212" data-btc="28577.5706649">$298,776,655</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="198868513.17">\n<span data-supply="198868513.17">\n<span data-supply-container>198,868,513</span>\n<span class="hidden-xs">TUSD</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.154663" data-symbol="TUSD" data-sort="0.154663">0.15%</td>\n<td><a href="/currencies/trueusd/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2563.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2563" data-cc-slug="trueusd">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2563" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2563">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/trueusd/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/trueusd/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/trueusd/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bitcoin-gold" class="">\n<td class="text-center">\n38\n</td>\n<td class="no-wrap currency-name" data-sort="Bitcoin Gold">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2083.png" class="logo-sprite lazyload" alt="Bitcoin Gold" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitcoin-gold/">BTG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitcoin-gold/">Bitcoin Gold</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="187799986.272" data-btc="17962.8069494" data-sort="187799986.272">\n$187,799,986\n</td>\n<td class="no-wrap text-right" data-sort="10.7228962898">\n<a href="/currencies/bitcoin-gold/#markets" class="price" data-usd="10.7228962898" data-btc="0.00102563008558">$10.72</a>\n</td>\n<td class="no-wrap text-right" data-sort="12819965.0876">\n<a href="/currencies/bitcoin-gold/#markets" class="volume" data-usd="12819965.0876" data-btc="1226.2117934">$12,819,965</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17513923.589">\n<span data-supply="17513923.589">\n<span data-supply-container>17,513,924</span>\n<span class="hidden-xs">BTG</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.401105" data-symbol="BTG" data-sort="0.401105">0.40%</td>\n<td><a href="/currencies/bitcoin-gold/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2083.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2083" data-cc-slug="bitcoin-gold">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2083" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n <ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2083">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-gold/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-gold/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-gold/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-zb" class="">\n<td class="text-center">\n39\n</td>\n<td class="no-wrap currency-name" data-sort="ZB">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3351.png" class="logo-sprite lazyload" alt="ZB" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/zb/">ZB</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/zb/">ZB</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="157291665.326" data-btc="15044.7285705" data-sort="157291665.326">\n$157,291,665\n</td>\n<td class="no-wrap text-right" data-sort="0.339511039186">\n<a href="/currencies/zb/#markets" class="price" data-usd="0.339511039186" data-btc="3.24737577204e-05">$0.339511</a>\n</td>\n<td class="no-wrap text-right" data-sort="75814496.9354">\n<a href="/currencies/zb/#markets" class="volume" data-usd="75814496.9354" data-btc="7251.55096893">$75,814,497</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="463288810.0">\n<span data-supply="463288810.0">\n<span data-supply-container>463,288,810</span>\n<span class="hidden-xs">ZB</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.621816" data-symbol="ZB" data-sort="0.621816">0.62%</td>\n<td><a href="/currencies/zb/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3351.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3351" data-cc-slug="zb">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3351" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3351">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zb/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zb/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zb/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-omisego" class="">\n<td class="text-center">\n40\n</td>\n<td class="no-wrap currency-name" data-sort="OmiseGO">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1808.png" class="logo-sprite lazyload" alt="OmiseGO" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/omisego/">OMG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/omisego/">OmiseGO</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="153276057.654" data-btc="14660.6412932" data-sort="153276057.654">\n$153,276,058\n</td>\n<td class="no-wrap text-right" data-sort="1.0929132761">\n<a href="/currencies/omisego/#markets" class="price" data-usd="1.0929132761" data-btc="0.000104535631662">$1.09</a>\n</td>\n<td class="no-wrap text-right" data-sort="41985316.4714">\n<a href="/currencies/omisego/#markets" class="volume" data-usd="41985316.4714" data-btc="4015.8370054">$41,985,316</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="140245398.245">\n<span data-supply="140245398.245">\n<span data-supply-container>140,245,398</span>\n<span class="hidden-xs">OMG</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.09926" data-symbol="OMG" data-sort="2.09926">2.10%</td>\n<td><a href="/currencies/omisego/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1808.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1808" data-cc-slug="omisego">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1808" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1808">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/omisego/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/omisego/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/omisego/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ravencoin" class="">\n<td class="text-center">\n41\n</td>\n<td class="no-wrap currency-name" data-sort="Ravencoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2577.png" class="logo-sprite lazyload" alt="Ravencoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ravencoin/">RVN</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ravencoin/">Ravencoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="140658522.993" data-btc="13453.791688" data-sort="140658522.993">\n$140,658,523\n</td>\n<td class="no-wrap text-right" data-sort="0.0321401245068">\n<a href="/currencies/ravencoin/#markets" class="price" data-usd="0.0321401245068" data-btc="3.07415811526e-06">$0.032140</a>\n</td>\n<td class="no-wrap text-right" data-sort="10945819.718">\n<a href="/currencies/ravencoin/#markets" class="volume" data-usd="10945819.718" data-btc="1046.9524008">$10,945,820</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="4376415000.0">\n <span data-supply="4376415000.0">\n<span data-supply-container>4,376,415,000</span>\n<span class="hidden-xs">RVN</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.98905" data-symbol="RVN" data-sort="-0.98905">-0.99%</td>\n<td><a href="/currencies/ravencoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2577.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2577" data-cc-slug="ravencoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2577" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2577">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ravencoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ravencoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ravencoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-kucoin-shares" class="">\n<td class="text-center">\n42\n</td>\n<td class="no-wrap currency-name" data-sort="KuCoin Shares">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2087.png" class="logo-sprite lazyload" alt="KuCoin Shares" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/kucoin-shares/">KCS</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/kucoin-shares/">KuCoin Shares</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="135749427.454" data-btc="12984.2435415" data-sort="135749427.454">\n$135,749,427\n</td>\n<td class="no-wrap text-right" data-sort="1.54108845753">\n<a href="/currencies/kucoin-shares/#markets" class="price" data-usd="1.54108845753" data-btc="0.000147402963143">$1.54</a>\n</td>\n<td class="no-wrap text-right" data-sort="18286478.3435">\n<a href="/currencies/kucoin-shares/#markets" class="volume" data-usd="18286478.3435" data-btc="1749.07616763">$18,286,478</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="88086720.0">\n<span data-supply="88086720.0">\n<span data-supply-container>88,086,720</span>\n<span class="hidden-xs">KCS</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="4.68579" data-symbol="KCS" data-sort="4.68579">4.69%</td>\n<td><a href="/currencies/kucoin-shares/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2087.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2087" data-cc-slug="kucoin-shares">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2087" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2087">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/kucoin-shares/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/kucoin-shares/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/kucoin-shares/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-lisk" class="">\n<td class="text-center">\n43\n</td>\n<td class="no-wrap currency-name" data-sort="Lisk">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1214.png" class="logo-sprite lazyload" alt="Lisk" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/lisk/">LSK</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/lisk/">Lisk</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="128104542.758" data-btc="12253.0209751" data-sort="128104542.758">\n$128,104,543\n</td>\n<td class="no-wrap text-right" data-sort="1.06780737804">\n<a href="/currencies/lisk/#markets" class="price" data-usd="1.06780737804" data-btc="0.000102134287503">$1.07</a>\n</td>\n<td class="no-wrap text-right" data-sort="1675220.56082">\n<a href="/currencies/lisk/#markets" class="volume" data-usd="1675220.56082" data-btc="160.232511882">$1,675,221</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="119969711.198">\n<span data-supply="119969711.198">\n<span data-supply-container>119,969,711</span>\n<span class="hidden-xs">LSK</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.763082" data-symbol="LSK" data-sort="-0.763082">-0.76%</td>\n<td><a href="/currencies/lisk/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1214.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1214" data-cc-slug="lisk">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1214" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1214">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lisk/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lisk/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lisk/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-nano" class="">\n<td class="text-center">\n44\n</td>\n<td class="no-wrap currency-name" data-sort="Nano">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1567.png" class="logo-sprite lazyload" alt="Nano" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/nano/">NANO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/nano/">Nano</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="123435417.186" data-btc="11806.4256214" data-sort="123435417.186">\n$123,435,417\n</td>\n<td class="no-wrap text-right" data-sort="0.926356432184">\n<a href="/currencies/nano/#markets" class="price" data-usd="0.926356432184" data-btc="8.8604701672e-05">$0.926356</a>\n</td>\n<td class="no-wrap text-right" data-sort="2527131.50207">\n<a href="/currencies/nano/#markets" class="volume" data-usd="2527131.50207" data-btc="241.716606101">$2,527,132</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="133248297.197">\n<span data-supply="133248297.197">\n<span data-supply-container>133,248,297</span>\n<span class="hidden-xs">NANO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.49808" data-symbol="NANO" data-sort="-0.49808">-0.50%</td>\n<td><a href="/currencies/nano/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1567.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1567" data-cc-slug="nano">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1567" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1567">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nano/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nano/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nano/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bittorrent" class="">\n<td class="text-center">\n45\n</td>\n<td class="no-wrap currency-name" data-sort="BitTorrent">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3718.png" class="logo-sprite lazyload" alt="BitTorrent" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bittorrent/">BTT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bittorrent/">BitTorrent</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="121628322.173" data-btc="11633.5795021" data-sort="121628322.173">\n$121,628,322\n</td>\n<td class="no-wrap text-right" data-sort="0.000573403399422">\n<a href="/currencies/bittorrent/#markets" class="price" data-usd="0.000573403399422" data-btc="5.48452360003e-08">$0.000573</a>\n</td>\n<td class="no-wrap text-right" data-sort="92919564.1135">\n<a href="/currencies/bittorrent/#markets" class="volume" data-usd="92919564.1135" data-btc="8887.62680513">$92,919,564</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="2.121165e+11">\n<span data-supply="2.121165e+11">\n<span data-supply-container>212,116,500,000</span>\n<span class="hidden-xs">BTT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="8.50179" data-symbol="BTT" data-sort="8.50179">8.50%</td>\n<td><a href="/currencies/bittorrent/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3718.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3718" data-cc-slug="bittorrent">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3718" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3718">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bittorrent/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bittorrent/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bittorrent/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-augur" class="">\n<td class="text-center">\n46\n</td>\n<td class="no-wrap currency-name" data-sort="Augur">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1104.png" class="logo-sprite lazyload" alt="Augur" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/augur/">REP</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/augur/">Augur</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="121563030.838" data-btc="11627.3344768" data-sort="121563030.838">\n$121,563,031\n</td>\n<td class="no-wrap text-right" data-sort="11.0511846216">\n<a href="/currencies/augur/#markets" class="price" data-usd="11.0511846216" data-btc="0.00105703040698">$11.05</a>\n</td>\n<td class="no-wrap text-right" data-sort="9888563.62435">\n<a href="/currencies/augur/#markets" class="volume" data-usd="9888563.62435" data-btc="945.827328942">$9,888,564</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="11000000.0">\n<span data-supply="11000000.0">\n<span data-supply-container>11,000,000</span>\n<span class="hidden-xs">REP</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.336742" data-symbol="REP" data-sort="0.336742">0.34%</td>\n<td><a href="/currencies/augur/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1104.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1104" data-cc-slug="augur">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1104" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1104">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/augur/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/augur/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/augur/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-algorand" class="">\n<td class="text-center">\n47\n</td>\n<td class="no-wrap currency-name" data-sort="Algorand">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/4030.png" class="logo-sprite lazyload" alt="Algorand" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/algorand/">ALGO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/algorand/">Algorand</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="115210396.17" data-btc="11019.7138245" data-sort="115210396.17">\n$115,210,396\n</td>\n<td class="no-wrap text-right" data-sort="0.369399493044">\n<a href="/currencies/algorand/#markets" class="price" data-usd="0.369399493044" data-btc="3.53325466763e-05">$0.369399</a>\n</td>\n<td class="no-wrap text-right" data-sort="49934122.2481">\n<a href="/currencies/algorand/#markets" class="volume" data-usd="49934122.2481" data-btc="4776.12919966">$49,934,122</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="311885636.931">\n<span data-supply="311885636.931">\n<span data-supply-container>311,885,637</span>\n<span class="hidden-xs">ALGO</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-1.5922" data-symbol="ALGO" data-sort="-1.5922">-1.59%</td>\n<td><a href="/currencies/algorand/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/4030.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="4030" data-cc-slug="algorand">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-4030" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-4030">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/algorand/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/algorand/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/algorand/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-insight-chain" class="">\n<td class="text-center">\n48\n</td>\n<td class="no-wrap currency-name" data-sort="Insight Chain">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3116.png" class="logo-sprite lazyload" alt="Insight Chain" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/insight-chain/">INB</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/insight-chain/">Insight Chain</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="115525609.61" data-btc="11049.8635507" data-sort="115525609.61">\n$115,525,610\n</td>\n<td class="no-wrap text-right" data-sort="0.330164966474">\n<a href="/currencies/insight-chain/#markets" class="price" data-usd="0.330164966474" data-btc="3.15798188911e-05">$0.330165</a>\n</td>\n<td class="no-wrap text-right" data-sort="9942102.37104">\n<a href="/currencies/insight-chain/#markets" class="volume" data-usd="9942102.37104" data-btc="950.948235446">$9,942,102</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="349902689.082">\n<span data-supply="349902689.082">\n<span data-supply-container>349,902,689</span>\n<span class="hidden-xs">INB</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.97633" data-symbol="INB" data-sort="1.97633">1.98%</td>\n<td><a href="/currencies/insight-chain/#charts">\n\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3116.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3116" data-cc-slug="insight-chain">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3116" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3116">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/insight-chain/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/insight-chain/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/insight-chain/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bitcoin-diamond" class="">\n<td class="text-center">\n49\n</td>\n<td class="no-wrap currency-name" data-sort="Bitcoin Diamond">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2222.png" class="logo-sprite lazyload" alt="Bitcoin Diamond" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitcoin-diamond/">BCD</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitcoin-diamond/">Bitcoin Diamond</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="112782333.331" data-btc="10787.472998" data-sort="112782333.331">\n$112,782,333\n</td>\n<td class="no-wrap text-right" data-sort="0.604754039266">\n<a href="/currencies/bitcoin-diamond/#markets" class="price" data-usd="0.604754039266" data-btc="5.78438810079e-05">$0.604754</a>\n</td>\n<td class="no-wrap text-right" data-sort="3346934.96469">\n<a href="/currencies/bitcoin-diamond/#markets" class="volume" data-usd="3346934.96469" data-btc="320.129664737">$3,346,935</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="186492897.953">\n<span data-supply="186492897.953">\n<span data-supply-container>186,492,898</span>\n<span class="hidden-xs">BCD</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-2.43459" data-symbol="BCD" data-sort="-2.43459">-2.43%</td>\n<td><a href="/currencies/bitcoin-diamond/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2222.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2222" data-cc-slug="bitcoin-diamond">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2222" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2222">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-diamond/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-diamond/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitcoin-diamond/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-holo" class="">\n<td class="text-center">\n50\n</td>\n<td class="no-wrap currency-name" data-sort="Holo">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2682.png" class="logo-sprite lazyload" alt="Holo" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/holo/">HOT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/holo/">Holo</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="108880566.294" data-btc="10414.2744188" data-sort="108880566.294">\n$108,880,566\n</td>\n<td class="no-wrap text-right" data-sort="0.000817332233851">\n<a href="/currencies/holo/#markets" class="price" data-usd="0.000817332233851" data-btc="7.81766890489e-08">$0.000817</a>\n</td>\n<td class="no-wrap text-right" data-sort="5693786.73364">\n<a href="/currencies/holo/#markets" class="volume" data-usd="5693786.73364" data-btc="544.602765622">$5,693,787</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1.33214575156e+11">\n<span data-supply="1.33214575156e+11">\n<span data-supply-container>133,214,575,156</span>\n<span class="hidden-xs">HOT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.53533" data-symbol="HOT" data-sort="1.53533">1.54%</td>\n<td><a href="/currencies/holo/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2682.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2682" data-cc-slug="holo">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2682" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2682">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/holo/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/holo/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/holo/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-waves" class="">\n<td class="text-center">\n51\n</td>\n<td class="no-wrap currency-name" data-sort="Waves">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1274.png" class="logo-sprite lazyload" alt="Waves" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/waves/">WAVES</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/waves/">Waves</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="106662732.365" data-btc="10202.1416945" data-sort="106662732.365">\n$106,662,732\n</td>\n<td class="no-wrap text-right" data-sort="1.06662732365">\n<a href="/currencies/waves/#markets" class="price" data-usd="1.06662732365" data-btc="0.000102021416945">$1.07</a>\n</td>\n<td class="no-wrap text-right" data-sort="6961774.18378">\n<a href="/currencies/waves/#markets" class="volume" data-usd="6961774.18378" data-btc="665.883998028">$6,961,774</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="100000000.0">\n<span data-supply="100000000.0">\n<span data-supply-container>100,000,000</span>\n<span class="hidden-xs">WAVES</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-1.08165" data-symbol="WAVES" data-sort="-1.08165">-1.08%</td>\n<td><a href="/currencies/waves/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1274.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1274" data-cc-slug="waves">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1274" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1274">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waves/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waves/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waves/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-digibyte" class="">\n<td class="text-center">\n52\n</td>\n<td class="no-wrap currency-name" data-sort="DigiByte">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/109.png" class="logo-sprite lazyload" alt="DigiByte" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/digibyte/">DGB</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/digibyte/">DigiByte</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="104766268.374" data-btc="10020.7475568" data-sort="104766268.374">\n$104,766,268\n</td>\n<td class="no-wrap text-right" data-sort="0.00855799808379">\n<a href="/currencies/digibyte/#markets" class="price" data-usd="0.00855799808379" data-btc="8.18560589402e-07">$0.008558</a>\n</td>\n<td class="no-wrap text-right" data-sort="3290352.70015">\n<a href="/currencies/digibyte/#markets" class="volume" data-usd="3290352.70015" data-btc="314.717649993">$3,290,353</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="12241913044.2">\n<span data-supply="12241913044.2">\n<span data-supply-container>12,241,913,044</span>\n<span class="hidden-xs">DGB</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="7.16191" data-symbol="DGB" data-sort="7.16191">7.16%</td>\n<td><a href="/currencies/digibyte/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/109.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="109" data-cc-slug="digibyte">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-109" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-109">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/digibyte/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/digibyte/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/digibyte/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-theta" class="">\n<td class="text-center">\n53\n</td>\n<td class="no-wrap currency-name" data-sort="THETA">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2416.png" class="logo-sprite lazyload" alt="THETA" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/theta/">THETA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/theta/">THETA</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="104287287.4" data-btc="9974.9336942" data-sort="104287287.4">\n$104,287,287\n</td>\n<td class="no-wrap text-right" data-sort="0.119801223589">\n<a href="/currencies/theta/#markets" class="price" data-usd="0.119801223589" data-btc="1.14588200689e-05">$0.119801</a>\n</td>\n<td class="no-wrap text-right" data-sort="628661.381932">\n<a href="/currencies/theta/#markets" class="volume" data-usd="628661.381932" data-btc="60.1305850142">$628,661</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="870502690.0">\n<span data-supply="870502690.0">\n<span data-supply-container>870,502,690</span>\n<span class="hidden-xs">THETA</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.418043" data-symbol="THETA" data-sort="-0.418043">-0.42%</td>\n<td><a href="/currencies/theta/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2416.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2416" data-cc-slug="theta">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2416" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2416">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/theta/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/theta/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/theta/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-hypercash" class="">\n<td class="text-center">\n54\n</td>\n<td class="no-wrap currency-name" data-sort="HyperCash">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1903.png" class="logo-sprite lazyload" alt="HyperCash" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/hypercash/">HC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/hypercash/">HyperCash</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="99604904.4602" data-btc="9527.0702919" data-sort="99604904.4602">\n$99,604,904\n</td>\n<td class="no-wrap text-right" data-sort="2.28820137572">\n<a href="/currencies/hypercash/#markets" class="price" data-usd="2.28820137572" data-btc="0.000218863272513">$2.29</a>\n</td>\n<td class="no-wrap text-right" data-sort="12014046.0274">\n<a href="/currencies/hypercash/#markets" class="volume" data-usd="12014046.0274" data-btc="1149.1267585">$12,014,046</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="43529780.8651">\n<span data-supply="43529780.8651">\n<span data-supply-container>43,529,781</span>\n<span class="hidden-xs">HC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="10.0494" data-symbol="HC" data-sort="10.0494">10.05%</td>\n<td><a href="/currencies/hypercash/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1903.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1903" data-cc-slug="hypercash">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1903" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1903">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hypercash/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hypercash/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/hypercash/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-icon" class="">\n<td class="text-center">\n55\n</td>\n<td class="no-wrap currency-name" data-sort="ICON">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2099.png" class="logo-sprite lazyload" alt="ICON" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/icon/">ICX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/icon/">ICON</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="99258821.5875" data-btc="9493.96794746" data-sort="99258821.5875">\n$99,258,822\n</td>\n<td class="no-wrap text-right" data-sort="0.201402831961">\n<a href="/currencies/icon/#markets" class="price" data-usd="0.201402831961" data-btc="1.92639001812e-05">$0.201403</a>\n</td>\n<td class="no-wrap text-right" data-sort="10755031.7347">\n<a href="/currencies/icon/#markets" class="volume" data-usd="10755031.7347" data-btc="1028.7037961">$10,755,032</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="492837268.578">\n<span data-supply="492837268.578">\n<span data-supply-container>492,837,269</span>\n<span class="hidden-xs">ICX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.15152" data-symbol="ICX" data-sort="-0.15152">-0.15%</td>\n<td><a href="/currencies/icon/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2099.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2099" data-cc-slug="icon">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2099" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n <ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2099">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/icon/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/icon/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/icon/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-0x" class="">\n<td class="text-center">\n56\n</td>\n<td class="no-wrap currency-name" data-sort="0x">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1896.png" class="logo-sprite lazyload" alt="0x" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/0x/">ZRX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/0x/">0x</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="97288527.1527" data-btc="9305.51203075" data-sort="97288527.1527">\n$97,288,527\n</td>\n<td class="no-wrap text-right" data-sort="0.162019049727">\n<a href="/currencies/0x/#markets" class="price" data-usd="0.162019049727" data-btc="1.54968962998e-05">$0.162019</a>\n</td>\n<td class="no-wrap text-right" data-sort="7057565.35208">\n<a href="/currencies/0x/#markets" class="volume" data-usd="7057565.35208" data-btc="675.046289772">$7,057,565</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="600475853.405">\n<span data-supply="600475853.405">\n<span data-supply-container>600,475,853</span>\n<span class="hidden-xs">ZRX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-1.32778" data-symbol="ZRX" data-sort="-1.32778">-1.33%</td>\n<td><a href="/currencies/0x/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1896.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1896" data-cc-slug="0x">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1896" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1896">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/0x/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/0x/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/0x/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bytecoin-bcn" class="">\n<td class="text-center">\n57\n</td>\n<td class="no-wrap currency-name" data-sort="Bytecoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/372.png" class="logo-sprite lazyload" alt="Bytecoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bytecoin-bcn/">BCN</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bytecoin-bcn/">Bytecoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="93179222.7419" data-btc="8912.46279104" data-sort="93179222.7419">\n$93,179,223\n</td>\n<td class="no-wrap text-right" data-sort="0.000506224958306">\n<a href="/currencies/bytecoin-bcn/#markets" class="price" data-usd="0.000506224958306" data-btc="4.84197117344e-08">$0.000506</a>\n</td>\n<td class="no-wrap text-right" data-sort="15469.2566685">\n<a href="/currencies/bytecoin-bcn/#markets" class="volume" data-usd="15469.2566685" data-btc="1.47961284078">$15,469</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1.84066828814e+11">\n<span data-supply="1.84066828814e+11">\n<span data-supply-container>184,066,828,814</span>\n<span class="hidden-xs">BCN</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.7475" data-symbol="BCN" data-sort="3.7475">3.75%</td>\n<td><a href="/currencies/bytecoin-bcn/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/372.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="372" data-cc-slug="bytecoin-bcn">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-372" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-372">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytecoin-bcn/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytecoin-bcn/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytecoin-bcn/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-maidsafecoin" class="">\n<td class="text-center">\n58\n</td>\n<td class="no-wrap currency-name" data-sort="MaidSafeCoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/291.png" class="logo-sprite lazyload" alt="MaidSafeCoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/maidsafecoin/">MAID</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/maidsafecoin/">MaidSafeCoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="92090729.3609" data-btc="8808.34991616" data-sort="92090729.3609">\n$92,090,729\n</td>\n<td class="no-wrap text-right" data-sort="0.203491854024">\n<a href="/currencies/maidsafecoin/#markets" class="price" data-usd="0.203491854024" data-btc="1.946371223e-05">$0.203492</a>\n</td>\n<td class="no-wrap text-right" data-sort="287020.408416">\n<a href="/currencies/maidsafecoin/#markets" class="volume" data-usd="287020.408416" data-btc="27.4531020436">$287,020</a>\n</td>\n <td class="no-wrap text-right circulating-supply" data-sort="452552412.0">\n<span data-supply="452552412.0">\n<span data-supply-container>452,552,412</span>\n<span class="hidden-xs">MAID</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="6.43489" data-symbol="MAID" data-sort="6.43489">6.43%</td>\n<td><a href="/currencies/maidsafecoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/291.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="291" data-cc-slug="maidsafecoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-291" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-291">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maidsafecoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maidsafecoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/maidsafecoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-lambda" class="">\n<td class="text-center">\n59\n</td>\n<td class="no-wrap currency-name" data-sort="Lambda">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3657.png" class="logo-sprite lazyload" alt="Lambda" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/lambda/">LAMB</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/lambda/">Lambda</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="92127410.9603" data-btc="8811.85846002" data-sort="92127410.9603">\n$92,127,411\n</td>\n<td class="no-wrap text-right" data-sort="0.146306783576">\n<a href="/currencies/lambda/#markets" class="price" data-usd="0.146306783576" data-btc="1.39940399407e-05">$0.146307</a>\n</td>\n<td class="no-wrap text-right" data-sort="51607733.965">\n<a href="/currencies/lambda/#markets" class="volume" data-usd="51607733.965" data-btc="4936.20782787">$51,607,734</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="629686530.649">\n<span data-supply="629686530.649">\n<span data-supply-container>629,686,531</span>\n<span class="hidden-xs">LAMB</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.20684" data-symbol="LAMB" data-sort="1.20684">1.21%</td>\n<td><a href="/currencies/lambda/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3657.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3657" data-cc-slug="lambda">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3657" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3657">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lambda/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lambda/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/lambda/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bitshares" class="">\n<td class="text-center">\n60\n</td>\n<td class="no-wrap currency-name" data-sort="BitShares">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/463.png" class="logo-sprite lazyload" alt="BitShares" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bitshares/">BTS</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bitshares/">BitShares</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="90032364.1055" data-btc="8611.47014825" data-sort="90032364.1055">\n$90,032,364\n</td>\n<td class="no-wrap text-right" data-sort="0.0328397101316">\n<a href="/currencies/bitshares/#markets" class="price" data-usd="0.0328397101316" data-btc="3.14107250526e-06">$0.032840</a>\n</td>\n<td class="no-wrap text-right" data-sort="2063017.61779">\n<a href="/currencies/bitshares/#markets" class="volume" data-usd="2063017.61779" data-btc="197.324759907">$2,063,018</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="2741570000.0">\n<span data-supply="2741570000.0">\n<span data-supply-container>2,741,570,000</span>\n<span class="hidden-xs">BTS</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.173221" data-symbol="BTS" data-sort="0.173221">0.17%</td>\n<td><a href="/currencies/bitshares/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/463.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="463" data-cc-slug="bitshares">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-463" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-463">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitshares/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitshares/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bitshares/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-pundi-x" class="">\n<td class="text-center">\n61\n</td>\n<td class="no-wrap currency-name" data-sort="Pundi X">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2603.png" class="logo-sprite lazyload" alt="Pundi X" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/pundi-x/">NPXS</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/pundi-x/">Pundi X</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="87495040.6695" data-btc="8368.77869789" data-sort="87495040.6695">\n$87,495,041\n</td>\n<td class="no-wrap text-right" data-sort="0.000371337303094">\n<a href="/currencies/pundi-x/#markets" class="price" data-usd="0.000371337303094" data-btc="3.55178955068e-08">$0.000371</a>\n</td>\n<td class="no-wrap text-right" data-sort="2151785.84705">\n<a href="/currencies/pundi-x/#markets" class="volume" data-usd="2151785.84705" data-btc="205.815317319">$2,151,786</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="2.35621468515e+11">\n<span data-supply="2.35621468515e+11">\n<span data-supply-container>235,621,468,515</span>\n<span class="hidden-xs">NPXS</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.61585" data-symbol="NPXS" data-sort="1.61585">1.62%</td>\n<td><a href="/currencies/pundi-x/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2603.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2603" data-cc-slug="pundi-x">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2603" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2603">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/pundi-x/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/pundi-x/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/pundi-x/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-komodo" class="">\n <td class="text-center">\n62\n</td>\n<td class="no-wrap currency-name" data-sort="Komodo">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1521.png" class="logo-sprite lazyload" alt="Komodo" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/komodo/">KMD</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/komodo/">Komodo</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="85419671.0314" data-btc="8170.27248445" data-sort="85419671.0314">\n$85,419,671\n</td>\n<td class="no-wrap text-right" data-sort="0.737442661104">\n<a href="/currencies/komodo/#markets" class="price" data-usd="0.737442661104" data-btc="7.05353627581e-05">$0.737443</a>\n</td>\n<td class="no-wrap text-right" data-sort="2554051.22726">\n<a href="/currencies/komodo/#markets" class="volume" data-usd="2554051.22726" data-btc="244.291440297">$2,554,051</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="115832288.443">\n<span data-supply="115832288.443">\n<span data-supply-container>115,832,288</span>\n<span class="hidden-xs">KMD</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-3.11501" data-symbol="KMD" data-sort="-3.11501">-3.12%</td>\n<td><a href="/currencies/komodo/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1521.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1521" data-cc-slug="komodo">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1521" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1521">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/komodo/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/komodo/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/komodo/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-iostoken" class="">\n<td class="text-center">\n63\n</td>\n<td class="no-wrap currency-name" data-sort="IOST">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2405.png" class="logo-sprite lazyload" alt="IOST" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/iostoken/">IOST</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/iostoken/">IOST</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="83658717.2683" data-btc="8001.83971125" data-sort="83658717.2683">\n$83,658,717\n</td>\n<td class="no-wrap text-right" data-sort="0.00696345569748">\n<a href="/currencies/iostoken/#markets" class="price" data-usd="0.00696345569748" data-btc="6.6604483247e-07">$0.006963</a>\n</td>\n<td class="no-wrap text-right" data-sort="14207575.6118">\n<a href="/currencies/iostoken/#markets" class="volume" data-usd="14207575.6118" data-btc="1358.93480612">$14,207,576</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="12013965608.8">\n<span data-supply="12013965608.8">\n<span data-supply-container>12,013,965,609</span>\n<span class="hidden-xs">IOST</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.76134" data-symbol="IOST" data-sort="-0.76134">-0.76%</td>\n<td><a href="/currencies/iostoken/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2405.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2405" data-cc-slug="iostoken">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2405" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2405">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iostoken/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iostoken/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/iostoken/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-nash-exchange" class="">\n<td class="text-center">\n64\n</td>\n<td class="no-wrap currency-name" data-sort="Nash Exchange">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3829.png" class="logo-sprite lazyload" alt="Nash Exchange" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/nash-exchange/">NEX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/nash-exchange/">Nash Exchange</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="81432336.02" data-btc="7788.88944777" data-sort="81432336.02">\n$81,432,336\n</td>\n<td class="no-wrap text-right" data-sort="2.24971849682">\n<a href="/currencies/nash-exchange/#markets" class="price" data-usd="2.24971849682" data-btc="0.00021518243878">$2.25</a>\n</td>\n<td class="no-wrap text-right" data-sort="3467031.45888">\n<a href="/currencies/nash-exchange/#markets" class="volume" data-usd="3467031.45888" data-btc="331.616727028">$3,467,031</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="36196678.0">\n<span data-supply="36196678.0">\n<span data-supply-container>36,196,678</span>\n<span class="hidden-xs">NEX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.15977" data-symbol="NEX" data-sort="2.15977">2.16%</td>\n<td><a href="/currencies/nash-exchange/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3829.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3829" data-cc-slug="nash-exchange">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3829" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3829">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nash-exchange/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nash-exchange/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nash-exchange/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-bytom" class="">\n<td class="text-center">\n65\n</td>\n<td class="no-wrap currency-name" data-sort="Bytom">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1866.png" class="logo-sprite lazyload" alt="Bytom" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/bytom/">BTM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/bytom/">Bytom</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="80573212.2015" data-btc="7706.7154519" data-sort="80573212.2015">\n$80,573,212\n</td>\n<td class="no-wrap text-right" data-sort="0.0803723396224">\n<a href="/currencies/bytom/#markets" class="price" data-usd="0.0803723396224" data-btc="7.68750226966e-06">$0.080372</a>\n</td>\n<td class="no-wrap text-right" data-sort="6431624.92573">\n<a href="/currencies/bytom/#markets" class="volume" data-usd="6431624.92573" data-btc="615.175960367">$6,431,625</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1002499275.0">\n<span data-supply="1002499275.0">\n<span data-supply-container>1,002,499,275</span>\n<span class="hidden-xs">BTM</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="3.86171" data-symbol="BTM" data-sort="3.86171">3.86%</td>\n<td><a href="/currencies/bytom/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1866.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1866" data-cc-slug="bytom">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1866" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1866">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytom/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytom/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/bytom/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-energi" class="">\n<td class="text-center">\n66\n</td>\n<td class="no-wrap currency-name" data-sort="Energi">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3218.png" class="logo-sprite lazyload" alt="Energi" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/energi/">NRG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/energi/">Energi</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="78704270.5367" data-btc="7527.95378642" data-sort="78704270.5367">\n$78,704,271\n</td>\n<td class="no-wrap text-right" data-sort="3.86930883536">\n<a href="/currencies/energi/#markets" class="price" data-usd="3.86930883536" data-btc="0.000370093997433">$3.87</a>\n</td>\n<td class="no-wrap text-right" data-sort="513334.470096">\n<a href="/currencies/energi/#markets" class="volume" data-usd="513334.470096" data-btc="49.0997266285">$513,334</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="20340653.56">\n<span data-supply="20340653.56">\n<span data-supply-container>20,340,654</span>\n<span class="hidden-xs">NRG</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-4.37822" data-symbol="NRG" data-sort="-4.37822">-4.38%</td>\n<td><a href="/currencies/energi/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3218.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3218" data-cc-slug="energi">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3218" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3218">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/energi/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/energi/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/energi/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-aurora" class="">\n<td class="text-center">\n67\n</td>\n<td class="no-wrap currency-name" data-sort="Aurora">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2874.png" class="logo-sprite lazyload" alt="Aurora" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/aurora/">AOA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/aurora/">Aurora</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="78579680.8964" data-btc="7516.03696092" data-sort="78579680.8964">\n$78,579,681\n</td>\n<td class="no-wrap text-right" data-sort="0.0120109623202">\n<a href="/currencies/aurora/#markets" class="price" data-usd="0.0120109623202" data-btc="1.1488318062e-06">$0.012011</a>\n</td>\n<td class="no-wrap text-right" data-sort="2935188.90069">\n<a href="/currencies/aurora/#markets" class="volume" data-usd="2935188.90069" data-btc="280.746727567">$2,935,189</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="6542330148.21">\n<span data-supply="6542330148.21">\n<span data-supply-container>6,542,330,148</span>\n<span class="hidden-xs">AOA</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-3.97135" data-symbol="AOA" data-sort="-3.97135">-3.97%</td>\n<td><a href="/currencies/aurora/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2874.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2874" data-cc-slug="aurora">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2874" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2874">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aurora/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aurora/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aurora/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-dai" class="">\n<td class="text-center">\n68\n</td>\n<td class="no-wrap currency-name" data-sort="Dai">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2308.png" class="logo-sprite lazyload" alt="Dai" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/dai/">DAI</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/dai/">Dai</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="78404681.8265" data-btc="7499.29854383" data-sort="78404681.8265">\n $78,404,682\n</td>\n<td class="no-wrap text-right" data-sort="1.00641329765">\n<a href="/currencies/dai/#markets" class="price" data-usd="1.00641329765" data-btc="9.62620292786e-05">$1.01</a>\n</td>\n<td class="no-wrap text-right" data-sort="24696397.0263">\n<a href="/currencies/dai/#markets" class="volume" data-usd="24696397.0263" data-btc="2362.17595611">$24,696,397</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="77905053.5298">\n<span data-supply="77905053.5298">\n<span data-supply-container>77,905,054</span>\n<span class="hidden-xs">DAI</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.447045" data-symbol="DAI" data-sort="0.447045">0.45%</td>\n<td><a href="/currencies/dai/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2308.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2308" data-cc-slug="dai">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2308" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2308">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dai/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dai/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dai/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-siacoin" class="">\n<td class="text-center">\n69\n</td>\n<td class="no-wrap currency-name" data-sort="Siacoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1042.png" class="logo-sprite lazyload" alt="Siacoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/siacoin/">SC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/siacoin/">Siacoin</a>\n</td>\n\n<td class="no-wrap market-cap text-right" data-usd="77401095.9301" data-btc="7403.30695151" data-sort="77401095.9301">\n$77,401,096\n</td>\n<td class="no-wrap text-right" data-sort="0.00185094597322">\n<a href="/currencies/siacoin/#markets" class="price" data-usd="0.00185094597322" data-btc="1.77040402668e-07">$0.001851</a>\n</td>\n<td class="no-wrap text-right" data-sort="1237922.35117">\n<a href="/currencies/siacoin/#markets" class="volume" data-usd="1237922.35117" data-btc="118.405547592">$1,237,922</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="41817047634.0">\n<span data-supply="41817047634.0">\n<span data-supply-container>41,817,047,634</span>\n<span class="hidden-xs">SC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.0540673" data-symbol="SC" data-sort="0.0540673">0.05%</td>\n<td><a href="/currencies/siacoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1042.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1042" data-cc-slug="siacoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1042" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1042">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/siacoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/siacoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/siacoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-monacoin" class="">\n<td class="text-center">\n70\n</td>\n<td class="no-wrap currency-name" data-sort="MonaCoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/213.png" class="logo-sprite lazyload" alt="MonaCoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/monacoin/">MONA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/monacoin/">MonaCoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="73935071.9556" data-btc="7071.78658899" data-sort="73935071.9556">\n$73,935,072\n</td>\n<td class="no-wrap text-right" data-sort="1.12483550391">\n<a href="/currencies/monacoin/#markets" class="price" data-usd="1.12483550391" data-btc="0.000107588948262">$1.12</a>\n</td>\n<td class="no-wrap text-right" data-sort="1381950.44657">\n<a href="/currencies/monacoin/#markets" class="volume" data-usd="1381950.44657" data-btc="132.181634185">$1,381,950</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="65729674.8712">\n<span data-supply="65729674.8712">\n<span data-supply-container>65,729,675</span>\n<span class="hidden-xs">MONA</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.0438811" data-symbol="MONA" data-sort="-0.0438811">-0.04%</td>\n<td><a href="/currencies/monacoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/213.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="213" data-cc-slug="monacoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-213" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-213">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monacoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monacoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/monacoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-verge" class="">\n<td class="text-center">\n71\n</td>\n<td class="no-wrap currency-name" data-sort="Verge">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/693.png" class="logo-sprite lazyload" alt="Verge" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/verge/">XVG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/verge/">Verge</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="71339214.1713" data-btc="6823.49640976" data-sort="71339214.1713">\n$71,339,214\n</td>\n<td class="no-wrap text-right" data-sort="0.00447924459414">\n<a href="/currencies/verge/#markets" class="price" data-usd="0.00447924459414" data-btc="4.2843350269e-07">$0.004479</a>\n</td>\n<td class="no-wrap text-right" data-sort="895104.391654">\n<a href="/currencies/verge/#markets" class="volume" data-usd="895104.391654" data-btc="85.6154875515">$895,104</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="15926617239.1">\n<span data-supply="15926617239.1">\n<span data-supply-container>15,926,617,239</span>\n<span class="hidden-xs">XVG</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.0806247" data-symbol="XVG" data-sort="-0.0806247">-0.08%</td>\n<td><a href="/currencies/verge/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/693.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="693" data-cc-slug="verge">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-693" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-693">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/verge/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/verge/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/verge/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-rif-token" class="">\n<td class="text-center">\n72\n</td>\n<td class="no-wrap currency-name" data-sort="RIF Token">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3701.png" class="logo-sprite lazyload" alt="RIF Token" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/rif-token/">RIF</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/rif-token/">RIF Token</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="68311552.1798" data-btc="6533.90475996" data-sort="68311552.1798">\n$68,311,552\n</td>\n<td class="no-wrap text-right" data-sort="0.142916890725">\n<a href="/currencies/rif-token/#markets" class="price" data-usd="0.142916890725" data-btc="1.36698014141e-05">$0.142917</a>\n</td>\n<td class="no-wrap text-right" data-sort="3868479.24809">\n<a href="/currencies/rif-token/#markets" class="volume" data-usd="3868479.24809" data-btc="370.014648567">$3,868,479</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="477980956.857">\n<span data-supply="477980956.857">\n<span data-supply-container>477,980,957</span>\n<span class="hidden-xs">RIF</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.127792" data-symbol="RIF" data-sort="0.127792">0.13%</td>\n<td><a href="/currencies/rif-token/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3701.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3701" data-cc-slug="rif-token">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3701" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3701">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/rif-token/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/rif-token/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/rif-token/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-quant" class="">\n<td class="text-center">\n73\n</td>\n<td class="no-wrap currency-name" data-sort="Quant">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3155.png" class="logo-sprite lazyload" alt="Quant" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/quant/">QNT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/quant/">Quant</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="66997434.4167" data-btc="6408.21122738" data-sort="66997434.4167">\n$66,997,434\n</td>\n<td class="no-wrap text-right" data-sort="5.5494813535">\n<a href="/currencies/quant/#markets" class="price" data-usd="5.5494813535" data-btc="0.000530800157129">$5.55</a>\n</td>\n<td class="no-wrap text-right" data-sort="2635354.98071">\n<a href="/currencies/quant/#markets" class="volume" data-usd="2635354.98071" data-btc="252.068031001">$2,635,355</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="12072738.0">\n<span data-supply="12072738.0">\n<span data-supply-container>12,072,738</span>\n<span class="hidden-xs">QNT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="5.31821" data-symbol="QNT" data-sort="5.31821">5.32%</td>\n<td><a href="/currencies/quant/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3155.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3155" data-cc-slug="quant">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3155" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3155">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/quant/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/quant/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/quant/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-enjin-coin" class="">\n<td class="text-center">\n74\n</td>\n<td class="no-wrap currency-name" data-sort="Enjin Coin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2130.png" class="logo-sprite lazyload" alt="Enjin Coin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/enjin-coin/">ENJ</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/enjin-coin/">Enjin Coin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="64166195.2856" data-btc="6137.40715041" data-sort="64166195.2856">\n$64,166,195\n</td>\n<td class="no-wrap text-right" data-sort="0.0826520532542">\n<a href="/currencies/enjin-coin/#markets" class="price" data-usd="0.0826520532542" data-btc="7.9055537013e-06">$0.082652</a>\n</td>\n<td class="no-wrap text-right" data-sort="4339917.23475">\n<a href="/currencies/enjin-coin/#markets" class="volume" data-usd="4339917.23475" data-btc="415.107034946">$4,339,917</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="776341213.064">\n<span data-supply="776341213.064">\n<span data-supply-container>776,341,213</span>\n<span class="hidden-xs">ENJ</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.0902" data-symbol="ENJ" data-sort="2.0902">2.09%</td>\n<td><a href="/currencies/enjin-coin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2130.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2130" data-cc-slug="enjin-coin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2130" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2130">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/enjin-coin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/enjin-coin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/enjin-coin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-zilliqa" class="">\n<td class="text-center">\n75\n</td>\n<td class="no-wrap currency-name" data-sort="Zilliqa">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2469.png" class="logo-sprite lazyload" alt="Zilliqa" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/zilliqa/">ZIL</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/zilliqa/">Zilliqa</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="60277786.4073" data-btc="5765.48626048" data-sort="60277786.4073">\n$60,277,786\n</td>\n<td class="no-wrap text-right" data-sort="0.00693856200321">\n<a href="/currencies/zilliqa/#markets" class="price" data-usd="0.00693856200321" data-btc="6.63663785308e-07">$0.006939</a>\n</td>\n<td class="no-wrap text-right" data-sort="7018556.51291">\n<a href="/currencies/zilliqa/#markets" class="volume" data-usd="7018556.51291" data-btc="671.315148672">$7,018,557</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="8687360058.09">\n<span data-supply="8687360058.09">\n<span data-supply-container>8,687,360,058</span>\n<span class="hidden-xs">ZIL</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.660985" data-symbol="ZIL" data-sort="0.660985">0.66%</td>\n<td><a href="/currencies/zilliqa/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2469.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2469" data-cc-slug="zilliqa">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2469" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2469">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zilliqa/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zilliqa/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zilliqa/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-aeternity" class="">\n<td class="text-center">\n76\n</td>\n<td class="no-wrap currency-name" data-sort="Aeternity">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1700.png" class="logo-sprite lazyload" alt="Aeternity" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/aeternity/">AE</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/aeternity/">Aeternity</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="58968430.7845" data-btc="5640.24822003" data-sort="58968430.7845">\n$58,968,431\n</td>\n<td class="no-wrap text-right" data-sort="0.20891116701">\n<a href="/currencies/aeternity/#markets" class="price" data-usd="0.20891116701" data-btc="1.99820619643e-05">$0.208911</a>\n</td>\n<td class="no-wrap text-right" data-sort="42508841.0363">\n<a href="/currencies/aeternity/#markets" class="volume" data-usd="42508841.0363" data-btc="4065.91140039">$42,508,841</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="282265575.5">\n<span data-supply="282265575.5">\n<span data-supply-container>282,265,576</span>\n<span class="hidden-xs">AE</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.95315" data-symbol="AE" data-sort="1.95315">1.95%</td>\n<td><a href="/currencies/aeternity/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1700.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1700" data-cc-slug="aeternity">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1700" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1700">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aeternity/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aeternity/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aeternity/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-steem" class="">\n<td class="text-center">\n77\n</td>\n<td class="no-wrap currency-name" data-sort="Steem">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1230.png" class="logo-sprite lazyload" alt="Steem" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/steem/">STEEM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/steem/">Steem</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="57768002.1598" data-btc="5525.42889514" data-sort="57768002.1598">\n$57,768,002\n</td>\n<td class="no-wrap text-right" data-sort="0.167974879928">\n<a href="/currencies/steem/#markets" class="price" data-usd="0.167974879928" data-btc="1.6066563158e-05">$0.167975</a>\n</td>\n<td class="no-wrap text-right" data-sort="463340.401562">\n<a href="/currencies/steem/#markets" class="volume" data-usd="463340.401562" data-btc="44.3178636502">$463,340</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="343908578.381">\n<span data-supply="343908578.381">\n<span data-supply-container>343,908,578</span>\n<span class="hidden-xs">STEEM</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.544935" data-symbol="STEEM" data-sort="-0.544935">-0.54%</td>\n<td><a href="/currencies/steem/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1230.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1230" data-cc-slug="steem">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1230" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1230">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/steem/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/steem/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/steem/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ardor" class="">\n<td class="text-center">\n78\n</td>\n<td class="no-wrap currency-name" data-sort="Ardor">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1320.png" class="logo-sprite lazyload" alt="Ardor" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ardor/">ARDR</a></span>\n <br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ardor/">Ardor</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="56469717.4811" data-btc="5401.24977502" data-sort="56469717.4811">\n$56,469,717\n</td>\n<td class="no-wrap text-right" data-sort="0.0565262722992">\n<a href="/currencies/ardor/#markets" class="price" data-usd="0.0565262722992" data-btc="5.40665916455e-06">$0.056526</a>\n</td>\n<td class="no-wrap text-right" data-sort="543043.297842">\n<a href="/currencies/ardor/#markets" class="volume" data-usd="543043.297842" data-btc="51.9413345971">$543,043</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="998999495.0">\n<span data-supply="998999495.0">\n<span data-supply-container>998,999,495</span>\n<span class="hidden-xs">ARDR</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.638512" data-symbol="ARDR" data-sort="0.638512">0.64%</td>\n<td><a href="/currencies/ardor/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1320.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1320" data-cc-slug="ardor">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1320" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1320">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ardor/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ardor/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ardor/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-golem-network-tokens" class="">\n<td class="text-center">\n79\n</td>\n<td class="no-wrap currency-name" data-sort="Golem">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1455.png" class="logo-sprite lazyload" alt="Golem" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/golem-network-tokens/">GNT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/golem-network-tokens/">Golem</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="56236718.0944" data-btc="5378.9637084" data-sort="56236718.0944">\n$56,236,718\n</td>\n<td class="no-wrap text-right" data-sort="0.0583096252728">\n<a href="/currencies/golem-network-tokens/#markets" class="price" data-usd="0.0583096252728" data-btc="5.577234391e-06">$0.058310</a>\n</td>\n<td class="no-wrap text-right" data-sort="1676235.24519">\n<a href="/currencies/golem-network-tokens/#markets" class="volume" data-usd="1676235.24519" data-btc="160.329565027">$1,676,235</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="964450000.0">\n<span data-supply="964450000.0">\n<span data-supply-container>964,450,000</span>\n<span class="hidden-xs">GNT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-0.686863" data-symbol="GNT" data-sort="-0.686863">-0.69%</td>\n<td><a href="/currencies/golem-network-tokens/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1455.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1455" data-cc-slug="golem-network-tokens">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1455" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1455">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/golem-network-tokens/#charts">View Chart</a></li>\n <li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/golem-network-tokens/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/golem-network-tokens/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-status" class="">\n<td class="text-center">\n80\n</td>\n<td class="no-wrap currency-name" data-sort="Status">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1759.png" class="logo-sprite lazyload" alt="Status" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/status/">SNT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/status/">Status</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="54539681.3791" data-btc="5216.64451174" data-sort="54539681.3791">\n$54,539,681\n</td>\n<td class="no-wrap text-right" data-sort="0.0157152963998">\n<a href="/currencies/status/#markets" class="price" data-usd="0.0157152963998" data-btc="1.50314619817e-06">$0.015715</a>\n</td>\n<td class="no-wrap text-right" data-sort="13800292.1798">\n<a href="/currencies/status/#markets" class="volume" data-usd="13800292.1798" data-btc="1319.97871348">$13,800,292</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="3470483788.0">\n<span data-supply="3470483788.0">\n<span data-supply-container>3,470,483,788</span>\n<span class="hidden-xs">SNT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.10305" data-symbol="SNT" data-sort="1.10305">1.10%</td>\n<td><a href="/currencies/status/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1759.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1759" data-cc-slug="status">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1759" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1759">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/status/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/status/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/status/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-crypto-com" class="">\n<td class="text-center">\n81\n</td>\n<td class="no-wrap currency-name" data-sort="Crypto.com">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1776.png" class="logo-sprite lazyload" alt="Crypto.com" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/crypto-com/">MCO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/crypto-com/">Crypto.com</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="50913066.0579" data-btc="4869.76381073" data-sort="50913066.0579">\n$50,913,066\n</td>\n<td class="no-wrap text-right" data-sort="3.22360456762">\n<a href="/currencies/crypto-com/#markets" class="price" data-usd="3.22360456762" data-btc="0.000308333284144">$3.22</a>\n</td>\n<td class="no-wrap text-right" data-sort="4111840.83033">\n<a href="/currencies/crypto-com/#markets" class="volume" data-usd="4111840.83033" data-btc="393.291844734">$4,111,841</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="15793831.095">\n<span data-supply="15793831.095">\n<span data-supply-container>15,793,831</span>\n<span class="hidden-xs">MCO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.214156" data-symbol="MCO" data-sort="0.214156">0.21%</td>\n<td><a href="/currencies/crypto-com/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1776.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1776" data-cc-slug="crypto-com">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1776" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1776">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/crypto-com/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-gxchain" class="">\n<td class="text-center">\n82\n</td>\n<td class="no-wrap currency-name" data-sort="GXChain">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1750.png" class="logo-sprite lazyload" alt="GXChain" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/gxchain/">GXC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/gxchain/">GXChain</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="50260219.5751" data-btc="4807.31995451" data-sort="50260219.5751">\n$50,260,220\n</td>\n<td class="no-wrap text-right" data-sort="0.773234147309">\n<a href="/currencies/gxchain/#markets" class="price" data-usd="0.773234147309" data-btc="7.39587685309e-05">$0.773234</a>\n</td>\n<td class="no-wrap text-right" data-sort="3013221.28089">\n<a href="/currencies/gxchain/#markets" class="volume" data-usd="3013221.28089" data-btc="288.210415979">$3,013,221</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="65000000.0">\n<span data-supply="65000000.0">\n<span data-supply-container>65,000,000</span>\n<span class="hidden-xs">GXC</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.78193" data-symbol="GXC" data-sort="2.78193">2.78%</td>\n <td><a href="/currencies/gxchain/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1750.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1750" data-cc-slug="gxchain">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1750" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1750">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/gxchain/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/gxchain/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/gxchain/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-wink-tronbet" class="">\n<td class="text-center">\n83\n</td>\n<td class="no-wrap currency-name" data-sort="WINk">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/4206.png" class="logo-sprite lazyload" alt="WINk" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/wink-tronbet/">WIN</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/wink-tronbet/">WINk</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="49455545.434" data-btc="4730.3539944" data-sort="49455545.434">\n$49,455,545\n</td>\n<td class="no-wrap text-right" data-sort="0.000251682019129">\n<a href="/currencies/wink-tronbet/#markets" class="price" data-usd="0.000251682019129" data-btc="2.40730343596e-08">$0.000252</a>\n</td>\n<td class="no-wrap text-right" data-sort="12692801.4921">\n<a href="/currencies/wink-tronbet/#markets" class="volume" data-usd="12692801.4921" data-btc="1214.04877271">$12,692,801</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="1.96500113934e+11">\n<span data-supply="1.96500113934e+11">\n<span data-supply-container>196,500,113,934</span>\n<span class="hidden-xs">WIN</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-4.09214" data-symbol="WIN" data-sort="-4.09214">-4.09%</td>\n<td><a href="/currencies/wink-tronbet/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/4206.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="4206" data-cc-slug="wink-tronbet">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-4206" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-4206">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wink-tronbet/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wink-tronbet/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wink-tronbet/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-synthetix-network-token" class="">\n<td class="text-center">\n84\n</td>\n<td class="no-wrap currency-name" data-sort="Synthetix Network Token">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2586.png" class="logo-sprite lazyload" alt="Synthetix Network Token" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/synthetix-network-token/">SNX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/synthetix-network-token/">Synthetix Net...</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="48701210.8283" data-btc="4658.20294066" data-sort="48701210.8283">\n$48,701,211\n</td>\n<td class="no-wrap text-right" data-sort="0.378638670332">\n<a href="/currencies/synthetix-network-token/#markets" class="price" data-usd="0.378638670332" data-btc="3.62162611071e-05">$0.378639</a>\n</td>\n<td class="no-wrap text-right" data-sort="17742.8750231">\n<a href="/currencies/synthetix-network-token/#markets" class="volume" data-usd="17742.8750231" data-btc="1.69708126765">$17,743</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="128621862.066">\n<span data-supply="128621862.066">\n<span data-supply-container>128,621,862</span>\n<span class="hidden-xs">SNX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="5.02304" data-symbol="SNX" data-sort="5.02304">5.02%</td>\n<td><a href="/currencies/synthetix-network-token/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2586.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2586" data-cc-slug="synthetix-network-token">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2586" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2586">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/synthetix-network-token/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/synthetix-network-token/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/synthetix-network-token/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-nexo" class="">\n<td class="text-center">\n85\n</td>\n<td class="no-wrap currency-name" data-sort="Nexo">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2694.png" class="logo-sprite lazyload" alt="Nexo" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/nexo/">NEXO</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/nexo/">Nexo</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="46707295.3807" data-btc="4467.48770703" data-sort="46707295.3807">\n$46,707,295\n</td>\n<td class="no-wrap text-right" data-sort="0.08340588297">\n<a href="/currencies/nexo/#markets" class="price" data-usd="0.08340588297" data-btc="7.977656463e-06">$0.083406</a>\n</td>\n<td class="no-wrap text-right" data-sort="7606683.80523">\n<a href="/currencies/nexo/#markets" class="volume" data-usd="7606683.80523" data-btc="727.568704507">$7,606,684</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="560000011.0">\n<span data-supply="560000011.0">\n<span data-supply-container>560,000,011</span>\n<span class="hidden-xs">NEXO</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-2.01887" data-symbol="NEXO" data-sort="-2.01887">-2.02%</td>\n<td><a href="/currencies/nexo/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2694.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2694" data-cc-slug="nexo">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2694" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2694">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nexo/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nexo/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/nexo/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-abbc-coin" class="">\n<td class="text-center">\n86\n</td>\n<td class="no-wrap currency-name" data-sort="ABBC Coin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3437.png" class="logo-sprite lazyload" alt="ABBC Coin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/abbc-coin/">ABBC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/abbc-coin/">ABBC Coin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="46524962.1042" data-btc="4450.04778327" data-sort="46524962.1042">\n$46,524,962\n</td>\n<td class="no-wrap text-right" data-sort="0.0839334487017">\n<a href="/currencies/abbc-coin/#markets" class="price" data-usd="0.0839334487017" data-btc="8.02811738997e-06">$0.083933</a>\n</td>\n<td class="no-wrap text-right" data-sort="33818793.7108">\n<a href="/currencies/abbc-coin/#markets" class="volume" data-usd="33818793.7108" data-btc="3234.72048506">$33,818,794</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="554307761.97">\n<span data-supply="554307761.97">\n<span data-supply-container>554,307,762</span>\n<span class="hidden-xs">ABBC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-6.41826" data-symbol="ABBC" data-sort="-6.41826">-6.42%</td>\n<td><a href="/currencies/abbc-coin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3437.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3437" data-cc-slug="abbc-coin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3437" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3437">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/abbc-coin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/abbc-coin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/abbc-coin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-waltonchain" class="">\n<td class="text-center">\n87\n</td>\n<td class="no-wrap currency-name" data-sort="Waltonchain">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1925.png" class="logo-sprite lazyload" alt="Waltonchain" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/waltonchain/">WTC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/waltonchain/">Waltonchain</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="45171738.4571" data-btc="4320.61382738" data-sort="45171738.4571">\n$45,171,738\n</td>\n<td class="no-wrap text-right" data-sort="1.06752087467">\n<a href="/currencies/waltonchain/#markets" class="price" data-usd="1.06752087467" data-btc="0.000102106883854">$1.07</a>\n</td>\n<td class="no-wrap text-right" data-sort="3016035.41505">\n<a href="/currencies/waltonchain/#markets" class="volume" data-usd="3016035.41505" data-btc="288.47958399">$3,016,035</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="42314618.4107">\n<span data-supply="42314618.4107">\n<span data-supply-container>42,314,618</span>\n<span class="hidden-xs">WTC</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.69493" data-symbol="WTC" data-sort="2.69493">2.69%</td>\n<td><a href="/currencies/waltonchain/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1925.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1925" data-cc-slug="waltonchain">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1925" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1925">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waltonchain/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waltonchain/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/waltonchain/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-metaverse" class="">\n<td class="text-center">\n88\n</td>\n<td class="no-wrap currency-name" data-sort="Metaverse ETP">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1703.png" class="logo-sprite lazyload" alt="Metaverse ETP" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/metaverse/">ETP</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/metaverse/">Metaverse ETP</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="44842055.7855" data-btc="4289.08013935" data-sort="44842055.7855">\n$44,842,056\n</td>\n<td class="no-wrap text-right" data-sort="0.588708504405">\n<a href="/currencies/metaverse/#markets" class="price" data-usd="0.588708504405" data-btc="5.63091479612e-05">$0.588709</a>\n</td>\n<td class="no-wrap text-right" data-sort="2814071.22861">\n<a href="/currencies/metaverse/#markets" class="volume" data-usd="2814071.22861" data-btc="269.161991035">$2,814,071</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="76170219.1321">\n<span data-supply="76170219.1321">\n<span data-supply-container>76,170,219</span>\n<span class="hidden-xs">ETP</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-8.44305" data-symbol="ETP" data-sort="-8.44305">-8.44%</td>\n<td><a href="/currencies/metaverse/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1703.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1703" data-cc-slug="metaverse">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1703" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1703">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/metaverse/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/metaverse/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/metaverse/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-zcoin" class="">\n<td class="text-center">\n89\n</td>\n<td class="no-wrap currency-name" data-sort="Zcoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1414.png" class="logo-sprite lazyload" alt="Zcoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/zcoin/">XZC</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/zcoin/">Zcoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="43584619.8708" data-btc="4168.80814659" data-sort="43584619.8708">\n$43,584,620\n</td>\n<td class="no-wrap text-right" data-sort="5.23349230812">\n<a href="/currencies/zcoin/#markets" class="price" data-usd="5.23349230812" data-btc="0.000500576245334">$5.23</a>\n</td>\n<td class="no-wrap text-right" data-sort="4088602.15848">\n<a href="/currencies/zcoin/#markets" class="volume" data-usd="4088602.15848" data-btc="391.069098159">$4,088,602</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="8328018.32978">\n<span data-supply="8328018.32978">\n<span data-supply-container>8,328,018</span>\n<span class="hidden-xs">XZC</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="4.15826" data-symbol="XZC" data-sort="4.15826">4.16%</td>\n<td><a href="/currencies/zcoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1414.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1414" data-cc-slug="zcoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1414" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1414">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/zcoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-xmax" class="">\n<td class="text-center">\n90\n</td>\n<td class="no-wrap currency-name" data-sort="XMax">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2859.png" class="logo-sprite lazyload" alt="XMax" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/xmax/">XMX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/xmax/">XMax</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="42784535.0701" data-btc="4092.28115049" data-sort="42784535.0701">\n$42,784,535\n</td>\n<td class="no-wrap text-right" data-sort="0.00251127747809">\n<a href="/currencies/xmax/#markets" class="price" data-usd="0.00251127747809" data-btc="2.40200190803e-07">$0.002511</a>\n</td>\n<td class="no-wrap text-right" data-sort="2567658.15345">\n<a href="/currencies/xmax/#markets" class="volume" data-usd="2567658.15345" data-btc="245.592923823">$2,567,658</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="17036960448.8">\n<span data-supply="17036960448.8">\n<span data-supply-container>17,036,960,449</span>\n<span class="hidden-xs">XMX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-7.05024" data-symbol="XMX" data-sort="-7.05024">-7.05%</td>\n <td><a href="/currencies/xmax/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2859.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2859" data-cc-slug="xmax">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2859" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2859">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/xmax/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/xmax/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/xmax/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-beam" class="">\n<td class="text-center">\n91\n</td>\n<td class="no-wrap currency-name" data-sort="Beam">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3702.png" class="logo-sprite lazyload" alt="Beam" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/beam/">BEAM</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/beam/">Beam</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="42330171.1484" data-btc="4048.82187465" data-sort="42330171.1484">\n$42,330,171\n</td>\n<td class="no-wrap text-right" data-sort="1.19213590994">\n<a href="/currencies/beam/#markets" class="price" data-usd="1.19213590994" data-btc="0.000114026138302">$1.19</a>\n</td>\n<td class="no-wrap text-right" data-sort="62403752.0755">\n<a href="/currencies/beam/#markets" class="volume" data-usd="62403752.0755" data-btc="5968.83191369">$62,403,752</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="35507840.0">\n<span data-supply="35507840.0">\n<span data-supply-container>35,507,840</span>\n<span class="hidden-xs">BEAM</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-1.09971" data-symbol="BEAM" data-sort="-1.09971">-1.10%</td>\n<td><a href="/currencies/beam/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3702.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3702" data-cc-slug="beam">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3702" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3702">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/beam/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/beam/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/beam/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-aelf" class="">\n<td class="text-center">\n92\n</td>\n<td class="no-wrap currency-name" data-sort="aelf">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2299.png" class="logo-sprite lazyload" alt="aelf" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/aelf/">ELF</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/aelf/">aelf</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="41635882.5087" data-btc="3982.41413388" data-sort="41635882.5087">\n$41,635,883\n</td>\n<td class="no-wrap text-right" data-sort="0.0833084207248">\n<a href="/currencies/aelf/#markets" class="price" data-usd="0.0833084207248" data-btc="7.96833433508e-06">$0.083308</a>\n</td>\n<td class="no-wrap text-right" data-sort="9144255.47714">\n<a href="/currencies/aelf/#markets" class="volume" data-usd="9144255.47714" data-btc="874.635292006">$9,144,255</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="499779999.986">\n<span data-supply="499779999.986">\n<span data-supply-container>499,780,000</span>\n<span class="hidden-xs">ELF</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="0.373628" data-symbol="ELF" data-sort="0.373628">0.37%</td>\n<td><a href="/currencies/aelf/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2299.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2299" data-cc-slug="aelf">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2299" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2299">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aelf/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aelf/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/aelf/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-grin" class="">\n<td class="text-center">\n93\n</td>\n<td class="no-wrap currency-name" data-sort="Grin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3709.png" class="logo-sprite lazyload" alt="Grin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/grin/">GRIN</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/grin/">Grin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="41630690.8733" data-btc="3981.91756118" data-sort="41630690.8733">\n$41,630,691\n</td>\n <td class="no-wrap text-right" data-sort="2.04701169149">\n<a href="/currencies/grin/#markets" class="price" data-usd="2.04701169149" data-btc="0.000195793815363">$2.05</a>\n</td>\n<td class="no-wrap text-right" data-sort="40644872.828">\n<a href="/currencies/grin/#markets" class="volume" data-usd="40644872.828" data-btc="3887.62543909">$40,644,873</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="20337300.0">\n<span data-supply="20337300.0">\n<span data-supply-container>20,337,300</span>\n<span class="hidden-xs">GRIN</span>\n</span>\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="1.77143" data-symbol="GRIN" data-sort="1.77143">1.77%</td>\n<td><a href="/currencies/grin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3709.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3709" data-cc-slug="grin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3709" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3709">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/grin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/grin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/grin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-solve" class="">\n<td class="text-center">\n94\n</td>\n<td class="no-wrap currency-name" data-sort="SOLVE">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/3724.png" class="logo-sprite lazyload" alt="SOLVE" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/solve/">SOLVE</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/solve/">SOLVE</a>\n</td>\n\n<td class="no-wrap market-cap text-right" data-usd="40318494.4977" data-btc="3856.40780666" data-sort="40318494.4977">\n$40,318,494\n</td>\n<td class="no-wrap text-right" data-sort="0.123145982936">\n<a href="/currencies/solve/#markets" class="price" data-usd="0.123145982936" data-btc="1.17787416388e-05">$0.123146</a>\n</td>\n<td class="no-wrap text-right" data-sort="984034.820374">\n<a href="/currencies/solve/#markets" class="volume" data-usd="984034.820374" data-btc="94.1215591159">$984,035</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="327404057.659">\n<span data-supply="327404057.659">\n<span data-supply-container>327,404,058</span>\n<span class="hidden-xs">SOLVE</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="5.73139" data-symbol="SOLVE" data-sort="5.73139">5.73%</td>\n<td><a href="/currencies/solve/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/3724.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="3724" data-cc-slug="solve">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-3724" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-3724">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/solve/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/solve/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/solve/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-stratis" class="">\n<td class="text-center">\n95\n</td>\n<td class="no-wrap currency-name" data-sort="Stratis">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/1343.png" class="logo-sprite lazyload" alt="Stratis" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/stratis/">STRAT</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/stratis/">Stratis</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="39264541.9866" data-btc="3755.59871788" data-sort="39264541.9866">\n$39,264,542\n</td>\n<td class="no-wrap text-right" data-sort="0.394705279478">\n<a href="/currencies/stratis/#markets" class="price" data-usd="0.394705279478" data-btc="3.77530098799e-05">$0.394705</a>\n</td>\n<td class="no-wrap text-right" data-sort="1175640.10108">\n<a href="/currencies/stratis/#markets" class="volume" data-usd="1175640.10108" data-btc="112.448337174">$1,175,640</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="99478127.17">\n<span data-supply="99478127.17">\n<span data-supply-container>99,478,127</span>\n<span class="hidden-xs">STRAT</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.74279" data-symbol="STRAT" data-sort="2.74279">2.74%</td>\n<td><a href="/currencies/stratis/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/1343.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="1343" data-cc-slug="stratis">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-1343" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-1343">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stratis/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stratis/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/stratis/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-reddcoin" class="">\n<td class="text-center">\n96\n</td>\n<td class="no-wrap currency-name" data-sort="ReddCoin">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/118.png" class="logo-sprite lazyload" alt="ReddCoin" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/reddcoin/">RDD</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/reddcoin/">ReddCoin</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="39112527.1699" data-btc="3741.05871252" data-sort="39112527.1699">\n$39,112,527\n</td>\n<td class="no-wrap text-right" data-sort="0.00135766311164">\n<a href="/currencies/reddcoin/#markets" class="price" data-usd="0.00135766311164" data-btc="1.29858584448e-07">$0.001358</a>\n</td>\n<td class="no-wrap text-right" data-sort="89169.5830629">\n<a href="/currencies/reddcoin/#markets" class="volume" data-usd="89169.5830629" data-btc="8.52894634399">$89,170</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="28808713173.8">\n<span data-supply="28808713173.8">\n<span data-supply-container>28,808,713,174</span>\n<span class="hidden-xs">RDD</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="2.94598" data-symbol="RDD" data-sort="2.94598">2.95%</td>\n<td><a href="/currencies/reddcoin/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/118.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="118" data-cc-slug="reddcoin">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-118" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-118">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/reddcoin/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/reddcoin/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/reddcoin/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-dragon-coins" class="">\n<td class="text-center">\n97\n</td>\n<td class="no-wrap currency-name" data-sort="Dragon Coins">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2593.png" class="logo-sprite lazyload" alt="Dragon Coins" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/dragon-coins/">DRG</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/dragon-coins/">Dragon Coins</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="38358163.6212" data-btc="3668.90489045" data-sort="38358163.6212">\n$38,358,164\n</td>\n<td class="no-wrap text-right" data-sort="0.110132068001">\n<a href="/currencies/dragon-coins/#markets" class="price" data-usd="0.110132068001" data-btc="1.05339788128e-05">$0.110132</a>\n</td>\n<td class="no-wrap text-right" data-sort="26319.9402286">\n<a href="/currencies/dragon-coins/#markets" class="volume" data-usd="26319.9402286" data-btc="2.51746560067">$26,320</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="348292412.169">\n<span data-supply="348292412.169">\n<span data-supply-container>348,292,412</span>\n<span class="hidden-xs">DRG</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  positive_change  text-right" data-timespan="24h" data-percentusd="4.28403" data-symbol="DRG" data-sort="4.28403">4.28%</td>\n<td><a href="/currencies/dragon-coins/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2593.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2593" data-cc-slug="dragon-coins">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2593" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2593">\n <li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dragon-coins/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dragon-coins/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/dragon-coins/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-wax" class="">\n<td class="text-center">\n98\n</td>\n<td class="no-wrap currency-name" data-sort="WAX">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2300.png" class="logo-sprite lazyload" alt="WAX" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/wax/">WAX</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/wax/">WAX</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="38062434.6236" data-btc="3640.618824" data-sort="38062434.6236">\n$38,062,435\n</td>\n<td class="no-wrap text-right" data-sort="0.0403707680569">\n<a href="/currencies/wax/#markets" class="price" data-usd="0.0403707680569" data-btc="3.86140769976e-06">$0.040371</a>\n</td>\n<td class="no-wrap text-right" data-sort="24011.0738251">\n<a href="/currencies/wax/#markets" class="volume" data-usd="24011.0738251" data-btc="2.29662574705">$24,011</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="942821661.702">\n<span data-supply="942821661.702">\n<span data-supply-container>942,821,662</span>\n<span class="hidden-xs">WAX</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-4.72247" data-symbol="WAX" data-sort="-4.72247">-4.72%</td>\n<td><a href="/currencies/wax/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2300.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2300" data-cc-slug="wax">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2300" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2300">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wax/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wax/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/wax/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-elastos" class="">\n<td class="text-center">\n99\n</td>\n<td class="no-wrap currency-name" data-sort="Elastos">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2492.png" class="logo-sprite lazyload" alt="Elastos" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/elastos/">ELA</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/elastos/">Elastos</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="37620094.7252" data-btc="3598.30962921" data-sort="37620094.7252">\n$37,620,095\n</td>\n<td class="no-wrap text-right" data-sort="2.35338225886">\n<a href="/currencies/elastos/#markets" class="price" data-usd="2.35338225886" data-btc="0.000225097733143">$2.35</a>\n</td>\n<td class="no-wrap text-right" data-sort="4285058.5223">\n<a href="/currencies/elastos/#markets" class="volume" data-usd="4285058.5223" data-btc="409.859875557">$4,285,059</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="15985543.608">\n<span data-supply="15985543.608">\n<span data-supply-container>15,985,544</span>\n<span class="hidden-xs">ELA</span>\n</span>\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-3.52347" data-symbol="ELA" data-sort="-3.52347">-3.52%</td>\n<td><a href="/currencies/elastos/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2492.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2492" data-cc-slug="elastos">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2492" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2492">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/elastos/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/elastos/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/elastos/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n<tr id="id-ren" class="">\n<td class="text-center">\n100\n</td>\n<td class="no-wrap currency-name" data-sort="Ren">\n<img data-src="https://s2.coinmarketcap.com/static/img/coins/32x32/2539.png" class="logo-sprite lazyload" alt="Ren" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" height="16" width="16">\n<span class="currency-symbol visible-xs"><a class="link-secondary" href="/currencies/ren/">REN</a></span>\n<br class="visible-xs">\n<a class="currency-name-container link-secondary" href="/currencies/ren/">Ren</a>\n</td>\n<td class="no-wrap market-cap text-right" data-usd="37570973.6026" data-btc="3593.61126228" data-sort="37570973.6026">\n$37,570,974\n</td>\n<td class="no-wrap text-right" data-sort="0.0472237468309">\n<a href="/currencies/ren/#markets" class="price" data-usd="0.0472237468309" data-btc="4.51688556846e-06">$0.047224</a>\n</td>\n<td class="no-wrap text-right" data-sort="5669907.2035">\n<a href="/currencies/ren/#markets" class="volume" data-usd="5669907.2035" data-btc="542.318721845">$5,669,907</a>\n</td>\n<td class="no-wrap text-right circulating-supply" data-sort="795594931.024">\n<span data-supply="795594931.024">\n<span data-supply-container>795,594,931</span>\n<span class="hidden-xs">REN</span>\n</span>\n*\n</td>\n<td class="no-wrap percent-change  negative_change text-right" data-timespan="24h" data-percentusd="-9.00574" data-symbol="REN" data-sort="-9.00574">-9.01%</td>\n<td><a href="/currencies/ren/#charts">\n<img class="sparkline lazyload" alt="sparkline" data-src="https://s2.coinmarketcap.com/generated/sparklines/web/7d/usd/2539.png" src="https://s2.coinmarketcap.com/static/cloud/img/loading-sparkline.svg" height="48" width="164">\n</a></td>\n<td class="dropdown" data-more-options data-cc-id="2539" data-cc-slug="ren">\n<button class="btn btn-transparent dropdown-toggle" type="button" id="dropdown-menu-2539" data-toggle="dropdown">\n<span class="glyphicons glyphicons-more text-gray"></span>\n</button>\n<ul class="dropdown-menu dropdown-menu-right" role="menu" aria-labelledby="dropdown-menu-2539">\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-add>Add to Watchlist</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-remove>Remove from Watchlist</a></li>\n<li class="disabled" role="presentation"><a role="menuitem" tabindex="-1" href="#" data-watchlist-full>Watchlist full!</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ren/#charts">View Chart</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ren/#markets">View Markets</a></li>\n<li role="presentation"><a role="menuitem" tabindex="-1" href="/currencies/ren/historical-data/">View Historical Data</a></li>\n</ul>\n</td>\n</tr>\n</tbody>\n</table>\n</div>\n<div class="pull-right">\n<ul class="pagination bottom-paginator">\n<li><a href="/2/">Next 100 &rarr;</a></li>\n<li>\n<a href="/all/views/all/">View All</a>\n</li>\n</ul>\n</div>\n<div id="asterisks">* Not Mineable</div>\n</div>\n</div>\n<div class="row text-center h3 margin-bottom--lv2">\n<strong>Total Market Cap: <span id="total-marketcap" class="market-cap" data-usd="2.67875759219e+11" data-btc="25637552.7306">$267,875,759,219\n</span></strong>\n</div>\n<div class="text-center text-gray">\nLast updated: Sep 08, 2019 11:46 PM UTC\n</div>\n</div>\n<div class="cmc-main-content__sidebar">\n\n<div id="div-gpt-ad-1542211140769-1" class="skyscraper">\n<script>googletag.cmd.push(function() { googletag.display(\'div-gpt-ad-1542211140769-1\'); });</script>\n</div>\n</div>\n</div>\n\n<div class="vertical-spacer-2x" id="newsletter"></div>\n<div class="cmc-newsletter-signup">\n<div class="cmc-newsletter-signup__title">Sign up for our newsletter</div>\n<div class="cmc-newsletter-signup__desc">Get crypto analysis, news and updates, right to your inbox! Sign up here so you don\'t miss a single one.</div>\n<script type="text/javascript" src="//app.sgwidget.com/js/sg-widget-v2.js"></script>\n<div class="sendgrid-subscription-widget widget-1037" data-emailerror="Please enter a valid email address" data-nameerror="Please enter your name" data-checkboxerror="Please tick the box to accept our conditions">\n<form class="sg-widget" data-token="cde39a92f0cf3a58f6915dfba2762a9d" onsubmit="return false;">\n<div class="sg-response"></div>\n<div class="cmc-newsletter-signup__email">\n<input class="sg_email" type="email" name="sg_email" placeholder="Enter your email..." required="required" aria-label="Email">\n</div>\n<div class="cmc-newsletter-signup__submit">\n<input type="submit" class="sg-submit-btn" id="widget-1037" value="Subscribe now">\n</div>\n</form>\n</div>\n</div>\n<div class="vertical-spacer-2x"></div>\n<div class="text-center">\n\n<div id="div-gpt-ad-1542211140769-2" class="responsive-leaderboard">\n<script>\n                                        googletag.cmd.push(function() { googletag.display(\'div-gpt-ad-1542211140769-2\'); });\n                                    </script>\n</div>\n</div>\n\n<div class="modal fade" id="donate_btc" tabindex="-1" role="dialog" aria-labelledby="donate_btc_label" aria-hidden="true">\n<div class="modal-dialog">\n<div class="modal-content">\n<div class="modal-header text-center">\n<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>\n<h4 class="modal-title" id="donate_btc_label">Donate Bitcoin</h4>\n</div>\n<div class="modal-body text-center">\n<strong>3CMCRgEm8HVz3DrWaCCid3vAANE42jcEv9</strong> <br /> <img src="https://s2.coinmarketcap.com/static/cloud/img/qrcodes/donate_bitcoin.png" alt="Donate Bitcoin"><br />\n</div>\n</div>\n</div>\n</div>\n<div class="modal fade" id="donate_ltc" tabindex="-1" role="dialog" aria-labelledby="donate_ltc_label" aria-hidden="true">\n<div class="modal-dialog">\n<div class="modal-content">\n<div class="modal-header text-center">\n<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>\n<h4 class="modal-title" id="donate_ltc_label">Donate Litecoin</h4>\n</div>\n<div class="modal-body text-center">\n<strong>LTdsVS8VDw6syvfQADdhf2PHAm3rMGJvPX</strong> <br /> <img src="https://s2.coinmarketcap.com/static/cloud/img/qrcodes/donate_litecoin.png" alt="Donate Litecoin"><br />\n</div>\n </div>\n</div>\n</div>\n<div class="modal fade" id="donate_eth" tabindex="-1" role="dialog" aria-labelledby="donate_eth_label" aria-hidden="true">\n<div class="modal-dialog">\n<div class="modal-content">\n<div class="modal-header text-center">\n<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>\n<h4 class="modal-title" id="donate_eth_label">Donate Ethereum</h4>\n</div>\n<div class="modal-body text-center">\n<strong>0x0074709077B8AE5a245E4ED161C971Dc4c3C8E2B</strong> <br /> <img src="https://s2.coinmarketcap.com/static/cloud/img/qrcodes/donate_ethereum.png" alt="Donate Ethereum"><br />\n</div>\n</div>\n</div>\n</div>\n<div class="modal fade" id="donate_bch" tabindex="-1" role="dialog" aria-labelledby="donate_bch_label" aria-hidden="true">\n<div class="modal-dialog">\n<div class="modal-content">\n<div class="modal-header text-center">\n<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>\n<h4 class="modal-title" id="donate_bch_label">Donate Bitcoin Cash</h4>\n</div>\n<div class="modal-body text-center">\n<strong>1LVXG4Z4oF6TrJfmUfSuLX8nqb8c5eCwha</strong> <br /> <img src="https://s2.coinmarketcap.com/static/cloud/img/qrcodes/donate_bitcoin_cash.png" alt="Donate Bitcoin Cash"><br />\n</div>\n</div>\n</div>\n</div>\n</div>\n<footer>\n<div class="container flex-container">\n<div class="footer-section footer-section--brand text-center">\n<div class="margin-bottom--lv1">\n<span class="cmc-logo cmc-logo_color_grey cmc-logo_size_md footer__cmc-logo"></span>\n</div>\n<div class="text-gray-light">&copy; 2019 CoinMarketCap</div>\n</div>\n<div class="footer-section footer-section--links">\n<section class="margin-bottom--lv2">\n<h5 class="footer-header">Useful Links</h5>\n<div class="row">\n<div class="col-xs-6">\n<ul class="list-unstyled">\n<li><a href="/advertising/" target="_blank" rel="noopener">Advertise</a></li>\n<li><a href="https://blockchain.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">Blockchain Explorer</a></li>\n<li><a href="/api/" target="_blank">Crypto API</a></li>\n<li><a href="/indices/" target="_blank">Crypto Indices</a></li>\n<li><a href="/disclaimer/">Disclaimer</a></li>\n<li><a href="/privacy/">Privacy</a></li>\n<li><a href="/terms/">Terms</a></li>\n<li><a href="/faq/">FAQ</a></li>\n<li><a href="/methodology/">Methodology</a></li>\n</ul>\n</div>\n<div class="col-xs-6">\n<ul class="list-unstyled">\n<li><a href="/request/" target="_blank">Request Form</a></li>\n<li><a href="/careers/">Careers</a></li>\n<li><a href="https://blog.coinmarketcap.com" target="_blank" rel="noopener">Blog</a></li>\n<li><a href="/#newsletter">Newsletter</a></li>\n<li><a href="https://shop.coinmarketcap.com?utm_source=coinmarketcap" target="_blank" rel="noopener">Shop</a></li>\n<li><a href="https://www.facebook.com/CoinMarketCap" target="_blank" rel="noopener">Facebook</a></li>\n<li><a href="https://twitter.com/CoinMarketCap" target="_blank" rel="noopener">Twitter</a></li>\n<li><a href="https://t.me/CoinMarketCap" target="_blank" rel="nofollow noopener">Telegram</a></li>\n<li><a href="https://t.me/CoinMarketCapBot" target="_blank" rel="nofollow noopener">Interactive Chat</a></li>\n</ul>\n</div>\n</div>\n</section>\n<section class="margin-bottom--lv2">\n<div class="row">\n<div class="col-xs-6">\n<h5 class="footer-header">Night Mode</h5>\n<div class="btn-group" data-toggle="buttons">\n<label class="btn btn-default btn-sm" data-night-mode-off>\n<input type="radio"> Off\n</label>\n<label class="btn btn-default btn-sm" data-night-mode-on>\n<input type="radio"> On\n</label>\n</div>\n</div>\n<div class="col-xs-6">\n<h5 class="footer-header">Downloads</h5>\n<ul class="list-unstyled footer-list--downloads">\n<li>\n<a href="https://itunes.apple.com/app/coinmarketcap/id1282107098?ls=1&mt=8" target="_blank" rel="nofollow noopener">\n<span class="app-store-badge"></span>\n</a>\n</li>\n<li>\n<a href=\'https://play.google.com/store/apps/details?id=com.coinmarketcap.android&hl=en_US\' target="_blank" rel="nofollow noopener">\n<span class="google-play-badge"></span>\n</a>\n</li>\n</ul>\n</div>\n</div>\n</section>\n</div>\n<div class="footer-section footer-section--donate overflow-ellipsis">\n<section class="margin-bottom--lv2">\n<h5 class="footer-header">Donate</h5>\n<ul class="list-unstyled">\n<li class="overflow-ellipsis">BTC: <a data-toggle="modal" data-target="#donate_btc">3CMCRgEm8HVz3DrWaCCid3vAANE42jcEv9</a></li>\n<li class="overflow-ellipsis">LTC: <a data-toggle="modal" data-target="#donate_ltc">LTdsVS8VDw6syvfQADdhf2PHAm3rMGJvPX</a></li>\n<li class="overflow-ellipsis">ETH: <a data-toggle="modal" data-target="#donate_eth">0x0074709077B8AE5a245E4ED161C971Dc4c3C8E2B</a></li>\n<li class="overflow-ellipsis">BCH: <a data-toggle="modal" data-target="#donate_bch">1LVXG4Z4oF6TrJfmUfSuLX8nqb8c5eCwha</a></li>\n</ul>\n</section>\n</div>\n</div>\n</footer>\n<div class="banner-alert banner-alert-fixed-bottom hide js-cookie-policy-banner">\n<div class="banner-alert-close">\n<button type="button" class="banner-alert-close-button js-close" aria-label="Close"><span aria-hidden="true">&times;</span></button>\n</div>\n<div class="banner-alert-body">\nWe use cookies to offer you a better browsing experience, analyze site traffic, personalize content, and serve targeted advertisements. Read about how we use cookies and how you can control them on our <a href="/privacy/">Privacy Policy</a>. If you continue to use this site, you consent to our use of cookies.\n</div>\n</div>\n<div id="currency-exchange-rates" data-pkr="0.00642456519362" data-all="0.00912719541794" data-ron="0.233367490555" data-mnt="0.00037569832927" data-ugx="0.000274401166842" data-ttd="0.149095845519" data-aud="0.684793610054" data-cup="0.0388349514563" data-ngn="0.00278605856295" data-byn="0.479413504695" data-gtq="0.131286489439" data-hrk="0.150085631357" data-clp="0.00140508330335" data-eur="1.10237782898" data-nok="0.111345679342" data-amd="0.00211629150938" data-mxn="0.0511733773984" data-gel="0.337837837838" data-try="0.175214966862" data-crc="0.00174379849237" data-aed="0.272256417628" data-gbp="1.22843331757" data-bam="0.569202825751" data-ars="0.0221100984735" data-mdl="0.0564166586884" data-btc="10445.8252469" data-sek="0.103714202654" data-mur="0.0276626611502" data-irr="2.37501484384e-05" data-cny="0.140534311452" data-bdt="0.0119425211711" data-pen="0.29993983207" data-kzt="0.00260202709568" data-rub="0.0151987002072" data-omr="2.60687537311" data-mkd="0.0181016037224" data-jod="1.4142051248" data-mmk="0.000661828551266" data-pln="0.254067493538" data-kes="0.00971345313259" data-hkd="0.127547506344" data-ves="4.48394989801e-05" data-cop="0.000295896744686" data-dzd="0.00837291930967" data-idr="7.09031286005e-05" data-krw="0.00083833440746" data-bgn="0.567556216443" data-isk="0.00791390118687" data-ils="0.284229713317" data-bob="0.145950969815" data-dkk="0.14773726346" data-twd="0.0320554420668" data-kwd="3.30328445573" data-brl="0.246135306485" data-zar="0.0675505566943" data-sgd="0.724003915413" data-uzs="0.000107181396986" data-usd="1.0" data-chf="1.01179551208" data-xau="1506.20556694" data-xpd="1535.48505973" data-xag="18.1868774405" data-huf="0.00334100480379" data-bmd="1.0" data-rsd="0.00945123117928" data-jmd="0.00735541708558" data-azn="0.587371512482" data-uah="0.0403020087448" data-lkr="0.00557909720712" data-tnd="0.348495197736" data-nzd="0.642793322663" data-iqd="0.000844693077674" data-xpt="946.96969697" data-cad="0.75934007273" data-lbp="0.000666601694333" data-khr="0.000245779533091" data-myr="0.239205836622" data-bhd="2.67174654744" data-inr="0.0139509463799" data-jpy="0.00934920204561" data-czk="0.0426373230924" data-sar="0.266581360631" data-ltc="70.8302111677" data-nad="0.0680561324258" data-qar="0.275153955517" data-pab="1.0" data-xrp="0.262908740378" data-vnd="4.30829739987e-05" data-uyu="0.0274847846294" data-nio="0.0300286097579" data-kgs="0.0143534313141" data-ssp="0.00767695378474" data-mad="0.104373826969" data-php="0.0192664996955" data-eth="181.757853128" data-npr="0.00881016982571" data-bch="308.497293126" data-ghs="0.183600373957" data-egp="0.0605987153072" data-hnl="0.0410820283216" data-dop="0.0196093692076" data-thb="0.0326157860404"></div>\n<div id="percentage_gains_data" data-bch1h="0.891453" data-eth1h="0.144714" data-xrp7d="2.25097" data-btc1h="0.00357045" data-bch7d="9.40004" data-xrp24h="0.679351" data-ltc7d="7.27343" data-xrp1h="-0.232852" data-bch24h="1.84339" data-ltc24h="2.35123" data-eth24h="1.65819" data-ltc1h="0.46121" data-btc24h="-0.752245" data-btc7d="7.45052" data-eth7d="6.05354"></div>\n<div class="scroll-to-top">\n<button class="btn"><span class="glyphicons glyphicons-chevron-up"></span></button>\n</div>\n<script>\n            var LANG_CODE = \'en\';\n            var STATIC_DOMAIN = \'s2.coinmarketcap.com\';\n            var CURRENCY_FLAGS = {\'USD\': \'us\', \'AUD\': \'au\', \'CHF\': \'ch\', \'IDR\': \'id\', \'KRW\': \'kr\', \'CNY\': \'cn\', \'TRY\': \'tr\', \'ILS\': \'il\', \'GBP\': \'gb\', \'NZD\': \'nz\', \'CLP\': \'cl\', \'DKK\': \'dk\', \'CAD\': \'ca\', \'PKR\': \'pk\', \'MXN\': \'mx\', \'HUF\': \'hu\', \'PHP\': \'ph\', \'TWD\': \'tw\', \'NOK\': \'no\', \'RUB\': \'ru\', \'SEK\': \'se\', \'MYR\': \'my\', \'INR\': \'in\', \'THB\': \'th\', \'JPY\': \'jp\', \'CZK\': \'cz\', \'BRL\': \'br\', \'PLN\': \'pl\', \'EUR\': \'eu\', \'ZAR\': \'za\', \'SGD\': \'sg\', \'HKD\': \'hk\'};\n        </script>\n<script src="https://s2.coinmarketcap.com/static/cloud/compressed/base.380c2015.min.js"></script>\n<script src="https://s2.coinmarketcap.com/static/cloud/compressed/currencies_main.373d0bab.min.js"></script>\n<script>\n        const PHRASES = {\n            emptyTable: "No data available in table",\n    zeroRecords: "No matching records found",\n    sortAscending: ": activate to sort column ascending",\n    sortDescending: ": activate to sort column descending",\n            \'K\': "K",\n    \'M\': "M",\n    \'B\': "B",\n    \'T\': "T"\n        };\n        var polyglot = new Polyglot({phrases: PHRASES});\n\n        var watchlistStore = WatchlistStore();\n        var watchlistActions = $.map($(\'[data-more-options]\'), function (elem) {\n            return new WatchlistAction({\n                element: elem,\n                watchlistStore: watchlistStore,\n                watchlistIsFull: watchlistStore.isFull()\n            });\n        });\n        $(document).on(watchlistStore.WATCHLIST_AT_CAPACITY, function (event) {\n            for (var i=0; i < watchlistActions.length; i++) {\n                watchlistActions[i].render({watchlistIsFull: true});\n            }\n        });\n        $(document).on(watchlistStore.WATCHLIST_UNDER_CAPACITY, function (event) {\n            for (var i=0; i < watchlistActions.length; i++) {\n                watchlistActions[i].render({watchlistIsFull: false});\n            }\n        });\n    </script>\n<script>\n            $( document ).ready(function() {\n                // GTM Session Set\n                var maxExpire = new Date(\'2038-01-19 00:00:00\');\n                if (Cookies.get("gtm_session_first") === undefined) {\n                    Cookies.set("gtm_session_first", new Date(), {expires: maxExpire})\n                }\n\n                Cookies.set("gtm_session_last", new Date(), {expires: maxExpire})\n\n                // Header Banner Prefs\n                $(".header-banner-close").on(\'click\', function () {\n                    Cookies.set("header-banner-noshow", "1", {expires: 1})\n                    $(\'#header-banner-wrapper\').fadeOut();\n                    dataLayer.push({\n                        \'event\': \'customEvent\',\n                        \'eventCategory\': \'Header Banner\',\n                        \'eventAction\': \'Close\',\n                    });\n                });\n\n                // Platform Alert Preferences\n                $(\'.js-platform-alert-closer\').on(\'click\', function () {\n                    var $alert = $(\'.js-platform-alert\');\n                    var alertId = $alert.data(\'id\');\n                    Cookies.set(\'cmc-hide-platform-alert-\' + alertId, \'1\', { expires: 7 });\n                    $alert.fadeOut();\n                    dataLayer.push({\n                        \'event\': \'customEvent\',\n                        \'eventCategory\': \'Platform Alert\',\n                        \'eventAction\': \'Dismissed\',\n                    });\n                });\n\n                $(\'.js-platform-alert a\').on(\'click\', function () {\n                    dataLayer.push({\n                        \'event\': \'customEvent\',\n                        \'eventCategory\': \'Platform Alert\',\n                        \'eventAction\': \'Link Clicked\',\n                    });\n                });\n\n            });\n\n            (function () {\n                NavMenu();\n\n                var polyglot = new Polyglot({phrases: {\n                    cryptocurrencies: "Cryptocurrencies",\n                    exchanges: "Exchanges"\n                }});\n\n                initQuickSearch({\n                    currencyDataURL: \'https://\' + STATIC_DOMAIN + \'/generated/search/quick_search.json\',\n                    exchangeDataURL: \'https://\' + STATIC_DOMAIN + \'/generated/search/quick_search_exchanges.json\',\n                    currencySuggestionTemplate: function (context) {\n                        var template = "/currencies/%25s/";\n                        context.url = template.replace(\'%25s\', context.slug);\n                        return Handlebars.templates[\'quick_search\'](context);\n                    },\n                    exchangeSuggestionTemplate: function (context) {\n                        var template = "/exchanges/%25s/";\n                        context.url = template.replace(\'%25s\', context.slug);\n                        return Handlebars.templates[\'quick_search_exchanges\'](context);\n                    },\n                    currencyLimit: 6,\n                    exchangeLimit: 4\n                }, polyglot);\n\n                $(\'[data-language-toggle]\').on(\'click\', function () {\n                    dataLayer.push({\n                        \'event\': \'customEvent\',\n                        \'eventCategory\': \'Language Dropdown\',\n                        \'eventAction\': \'Language Selected\',\n                        \'eventLabel\': $(this).data(\'language-toggle\')\n                    });\n                });\n\n                var language = $(\'[data-language-toggle="\' + \'en\' + \'"]\').html();\n                $(\'[data-language-dropdown]\').html(language);\n\n                NightMode.initSwitch();\n                ThemeSwitch(\'.js-theme-switch\');\n                ScrollToTop(\'.scroll-to-top\');\n                CookiedDismissableAlert({\n                    selector: \'.js-cookie-policy-banner\',\n                    cookieKey: \'cmc_gdpr_hide\'\n                });\n            })();\n\n            if (\'serviceWorker\' in navigator) {\n                navigator.serviceWorker.register(\'/sw.js\', { scope: \'/\' });\n            }\n        </script>\n</body>\n</html>'




```python
from bs4 import BeautifulSoup
```


```python
s = BeautifulSoup(m.content)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-1438e24122f1> in <module>
    ----> 1 s = BeautifulSoup(m.content)
    

    NameError: name 'm' is not defined



```python

```


```python

```


```python
coinNames = s.find_all('span', attrs={'class': 'currency-symbol'})
coinNames = [x.text for x in coinNames]
fullName = s.find_all('a', attrs={'class':'currency-name-container'})
fullName = [x.text for x in fullName]
MarketCap = s.find_all('td', attrs={'class':'no-wrap market-cap text-right'})
MarketCap = [x.text.split('\n')[1].split('$')[1] for x in MarketCap]
Price = s.find_all('a', attrs={'class':'price'})
Price = [x.text for x in Price]
```


```python
import pandas as pd
results = pd.DataFrame(columns = ['Prefix', 'Coin Name', 'Market Cap', 'Current Price'])
results['Prefix'] = coinNames
results['Coin Name'] = fullName
results['Market Cap'] = MarketCap
results['Current Price'] = Price
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prefix</th>
      <th>Coin Name</th>
      <th>Market Cap</th>
      <th>Current Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>187,230,971,725</td>
      <td>$10445.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ETH</td>
      <td>Ethereum</td>
      <td>19,568,558,583</td>
      <td>$181.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XRP</td>
      <td>XRP</td>
      <td>11,301,041,802</td>
      <td>$0.262909</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BCH</td>
      <td>Bitcoin Cash</td>
      <td>5,550,641,403</td>
      <td>$308.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LTC</td>
      <td>Litecoin</td>
      <td>4,477,350,589</td>
      <td>$70.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USDT</td>
      <td>Tether</td>
      <td>4,091,812,877</td>
      <td>$1.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BNB</td>
      <td>Binance Coin</td>
      <td>3,512,633,215</td>
      <td>$22.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EOS</td>
      <td>EOS</td>
      <td>3,511,065,748</td>
      <td>$3.77</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BSV</td>
      <td>Bitcoin SV</td>
      <td>2,420,008,478</td>
      <td>$135.54</td>
    </tr>
    <tr>
      <th>9</th>
      <td>XMR</td>
      <td>Monero</td>
      <td>1,332,090,844</td>
      <td>$77.45</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ADA</td>
      <td>Cardano</td>
      <td>1,212,509,111</td>
      <td>$0.046766</td>
    </tr>
    <tr>
      <th>11</th>
      <td>XLM</td>
      <td>Stellar</td>
      <td>1,180,805,534</td>
      <td>$0.059795</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LEO</td>
      <td>UNUS SED LEO</td>
      <td>1,069,184,894</td>
      <td>$1.07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TRX</td>
      <td>TRON</td>
      <td>1,056,225,131</td>
      <td>$0.015840</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HT</td>
      <td>Huobi Token</td>
      <td>998,567,383</td>
      <td>$4.06</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DASH</td>
      <td>Dash</td>
      <td>793,031,755</td>
      <td>$87.84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ETC</td>
      <td>Ethereum Classic</td>
      <td>755,620,188</td>
      <td>$6.67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>XTZ</td>
      <td>Tezos</td>
      <td>702,942,628</td>
      <td>$1.06</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MIOTA</td>
      <td>IOTA</td>
      <td>680,830,610</td>
      <td>$0.244944</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NEO</td>
      <td>NEO</td>
      <td>657,714,003</td>
      <td>$9.32</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LINK</td>
      <td>Chainlink</td>
      <td>644,019,550</td>
      <td>$1.84</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ATOM</td>
      <td>Cosmos</td>
      <td>521,258,265</td>
      <td>$2.73</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MKR</td>
      <td>Maker</td>
      <td>448,506,566</td>
      <td>$448.51</td>
    </tr>
    <tr>
      <th>23</th>
      <td>USDC</td>
      <td>USD Coin</td>
      <td>443,613,230</td>
      <td>$1.00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>XEM</td>
      <td>NEM</td>
      <td>424,022,836</td>
      <td>$0.047114</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ONT</td>
      <td>Ontology</td>
      <td>405,590,354</td>
      <td>$0.760410</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CRO</td>
      <td>Crypto.com Chain</td>
      <td>381,468,608</td>
      <td>$0.039472</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ZEC</td>
      <td>Zcash</td>
      <td>357,609,760</td>
      <td>$48.51</td>
    </tr>
    <tr>
      <th>28</th>
      <td>DOGE</td>
      <td>Dogecoin</td>
      <td>303,857,600</td>
      <td>$0.002509</td>
    </tr>
    <tr>
      <th>29</th>
      <td>VSYS</td>
      <td>V Systems</td>
      <td>274,208,570</td>
      <td>$0.152375</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>XVG</td>
      <td>Verge</td>
      <td>71,339,214</td>
      <td>$0.004479</td>
    </tr>
    <tr>
      <th>71</th>
      <td>RIF</td>
      <td>RIF Token</td>
      <td>68,311,552</td>
      <td>$0.142917</td>
    </tr>
    <tr>
      <th>72</th>
      <td>QNT</td>
      <td>Quant</td>
      <td>66,997,434</td>
      <td>$5.55</td>
    </tr>
    <tr>
      <th>73</th>
      <td>ENJ</td>
      <td>Enjin Coin</td>
      <td>64,166,195</td>
      <td>$0.082652</td>
    </tr>
    <tr>
      <th>74</th>
      <td>ZIL</td>
      <td>Zilliqa</td>
      <td>60,277,786</td>
      <td>$0.006939</td>
    </tr>
    <tr>
      <th>75</th>
      <td>AE</td>
      <td>Aeternity</td>
      <td>58,968,431</td>
      <td>$0.208911</td>
    </tr>
    <tr>
      <th>76</th>
      <td>STEEM</td>
      <td>Steem</td>
      <td>57,768,002</td>
      <td>$0.167975</td>
    </tr>
    <tr>
      <th>77</th>
      <td>ARDR</td>
      <td>Ardor</td>
      <td>56,469,717</td>
      <td>$0.056526</td>
    </tr>
    <tr>
      <th>78</th>
      <td>GNT</td>
      <td>Golem</td>
      <td>56,236,718</td>
      <td>$0.058310</td>
    </tr>
    <tr>
      <th>79</th>
      <td>SNT</td>
      <td>Status</td>
      <td>54,539,681</td>
      <td>$0.015715</td>
    </tr>
    <tr>
      <th>80</th>
      <td>MCO</td>
      <td>Crypto.com</td>
      <td>50,913,066</td>
      <td>$3.22</td>
    </tr>
    <tr>
      <th>81</th>
      <td>GXC</td>
      <td>GXChain</td>
      <td>50,260,220</td>
      <td>$0.773234</td>
    </tr>
    <tr>
      <th>82</th>
      <td>WIN</td>
      <td>WINk</td>
      <td>49,455,545</td>
      <td>$0.000252</td>
    </tr>
    <tr>
      <th>83</th>
      <td>SNX</td>
      <td>Synthetix Net...</td>
      <td>48,701,211</td>
      <td>$0.378639</td>
    </tr>
    <tr>
      <th>84</th>
      <td>NEXO</td>
      <td>Nexo</td>
      <td>46,707,295</td>
      <td>$0.083406</td>
    </tr>
    <tr>
      <th>85</th>
      <td>ABBC</td>
      <td>ABBC Coin</td>
      <td>46,524,962</td>
      <td>$0.083933</td>
    </tr>
    <tr>
      <th>86</th>
      <td>WTC</td>
      <td>Waltonchain</td>
      <td>45,171,738</td>
      <td>$1.07</td>
    </tr>
    <tr>
      <th>87</th>
      <td>ETP</td>
      <td>Metaverse ETP</td>
      <td>44,842,056</td>
      <td>$0.588709</td>
    </tr>
    <tr>
      <th>88</th>
      <td>XZC</td>
      <td>Zcoin</td>
      <td>43,584,620</td>
      <td>$5.23</td>
    </tr>
    <tr>
      <th>89</th>
      <td>XMX</td>
      <td>XMax</td>
      <td>42,784,535</td>
      <td>$0.002511</td>
    </tr>
    <tr>
      <th>90</th>
      <td>BEAM</td>
      <td>Beam</td>
      <td>42,330,171</td>
      <td>$1.19</td>
    </tr>
    <tr>
      <th>91</th>
      <td>ELF</td>
      <td>aelf</td>
      <td>41,635,883</td>
      <td>$0.083308</td>
    </tr>
    <tr>
      <th>92</th>
      <td>GRIN</td>
      <td>Grin</td>
      <td>41,630,691</td>
      <td>$2.05</td>
    </tr>
    <tr>
      <th>93</th>
      <td>SOLVE</td>
      <td>SOLVE</td>
      <td>40,318,494</td>
      <td>$0.123146</td>
    </tr>
    <tr>
      <th>94</th>
      <td>STRAT</td>
      <td>Stratis</td>
      <td>39,264,542</td>
      <td>$0.394705</td>
    </tr>
    <tr>
      <th>95</th>
      <td>RDD</td>
      <td>ReddCoin</td>
      <td>39,112,527</td>
      <td>$0.001358</td>
    </tr>
    <tr>
      <th>96</th>
      <td>DRG</td>
      <td>Dragon Coins</td>
      <td>38,358,164</td>
      <td>$0.110132</td>
    </tr>
    <tr>
      <th>97</th>
      <td>WAX</td>
      <td>WAX</td>
      <td>38,062,435</td>
      <td>$0.040371</td>
    </tr>
    <tr>
      <th>98</th>
      <td>ELA</td>
      <td>Elastos</td>
      <td>37,620,095</td>
      <td>$2.35</td>
    </tr>
    <tr>
      <th>99</th>
      <td>REN</td>
      <td>Ren</td>
      <td>37,570,974</td>
      <td>$0.047224</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>




```python
import datetime as dt
time=dt.datetime.now()
filename = str(time.year) + str(time.month) + str(time.day) + 'at' + str(time.hour) + '.csv'
results.to_csv(filename, index=False)
```


```python


```


```python

```

### <font color="green">Q: What part of the webpage do we want?</font>


```python

```


```python
from bs4 import BeautifulSoup
```

### <font color="green">Q: What is the equivalent of a BeautifulSoup object?</font>


```python

```


```python

```


```python
coinNames = s.find_all('span', attrs={'class': 'currency-symbol'})
coinNames = [x.text for x in coinNames]

fullName = s.find_all('a', attrs={'class':'currency-name-container'})
fullName = [x.text for x in fullName]

MarketCap = s.find_all('td', attrs={'class':'no-wrap market-cap text-right'})
MarketCap = [x.text.split('\n')[1].split('$')[1] for x in MarketCap]    


```


```python
import pandas as pd
results = pd.DataFrame(columns = ['Prefix', 'Coin Name', 'Market Cap', 'Current Price'])

```


```python
import datetime as dt

```

## 3.2 API Pulls
“An application program interface (API) is code that allows two software programs to communicate with each other. The API defines the correct way for a developer to write a program that requests services from an operating system (OS) or other application.” — TechTarget

API is actually a very simple tool that allows anyone to access information from a given website.

http://www.pythonforbeginners.com/api/list-of-python-apis

### <font color="green">Q: Which API is your most interested API listed there so far?</font>


```python

```


```python
!pip install coinmarketcap
```

    Collecting coinmarketcap
      Downloading https://files.pythonhosted.org/packages/a8/da/c64662a91905017f237f5ff2778b68638946b0d6268513efadd4d1363669/coinmarketcap-5.0.3.tar.gz
    Requirement already satisfied: requests>=2.18.4 in ./anaconda3/lib/python3.7/site-packages (from coinmarketcap) (2.22.0)
    Collecting requests_cache>=0.4.13 (from coinmarketcap)
      Downloading https://files.pythonhosted.org/packages/7f/55/9b1c40eb83c16d8fc79c5f6c2ffade04208b080670fbfc35e0a5effb5a92/requests_cache-0.5.2-py2.py3-none-any.whl
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in ./anaconda3/lib/python3.7/site-packages (from requests>=2.18.4->coinmarketcap) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in ./anaconda3/lib/python3.7/site-packages (from requests>=2.18.4->coinmarketcap) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in ./anaconda3/lib/python3.7/site-packages (from requests>=2.18.4->coinmarketcap) (1.24.2)
    Requirement already satisfied: certifi>=2017.4.17 in ./anaconda3/lib/python3.7/site-packages (from requests>=2.18.4->coinmarketcap) (2019.6.16)
    Building wheels for collected packages: coinmarketcap
      Building wheel for coinmarketcap (setup.py) ... [?25ldone
    [?25h  Stored in directory: /Users/richardmontero/Library/Caches/pip/wheels/5c/73/ec/47f4d3160b8d215cc223937a3886eccfc690cd3dbb5152ab42
    Successfully built coinmarketcap
    Installing collected packages: requests-cache, coinmarketcap
    Successfully installed coinmarketcap-5.0.3 requests-cache-0.5.2



```python
from coinmarketcap import Market

```


```python
cap = Market()
cap.ticker(start=0, limit=100)
```




    {'attention': 'WARNING: This API is now deprecated and will be taken offline soon.  Please switch to the new CoinMarketCap API to avoid interruptions in service. (https://pro.coinmarketcap.com/migrate/)',
     'data': {'1': {'id': 1,
       'name': 'Bitcoin',
       'symbol': 'BTC',
       'website_slug': 'bitcoin',
       'rank': 1,
       'circulating_supply': 17924062.0,
       'total_supply': 17924062.0,
       'max_supply': 21000000.0,
       'quotes': {'USD': {'price': 10443.3301176,
         'volume_24h': 13648509947.8822,
         'market_cap': 187186896514.0,
         'percent_change_1h': 0.01,
         'percent_change_24h': -0.79,
         'percent_change_7d': 7.4}},
       'last_updated': 1567986995},
      '1027': {'id': 1027,
       'name': 'Ethereum',
       'symbol': 'ETH',
       'website_slug': 'ethereum',
       'rank': 2,
       'circulating_supply': 107662984.0,
       'total_supply': 107662984.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 181.140264509,
         'volume_24h': 6455797234.88394,
         'market_cap': 19502101343.0,
         'percent_change_1h': -0.19,
         'percent_change_24h': 1.31,
         'percent_change_7d': 5.68}},
       'last_updated': 1567986984},
      '52': {'id': 52,
       'name': 'XRP',
       'symbol': 'XRP',
       'website_slug': 'ripple',
       'rank': 3,
       'circulating_supply': 42984656144.0,
       'total_supply': 99991362294.0,
       'max_supply': 100000000000.0,
       'quotes': {'USD': {'price': 0.263079672,
         'volume_24h': 1043368394.08535,
         'market_cap': 11308389239.0,
         'percent_change_1h': -0.16,
         'percent_change_24h': 0.74,
         'percent_change_7d': 2.31}},
       'last_updated': 1567986966},
      '1831': {'id': 1831,
       'name': 'Bitcoin Cash',
       'symbol': 'BCH',
       'website_slug': 'bitcoin-cash',
       'rank': 4,
       'circulating_supply': 17992513.0,
       'total_supply': 17992513.0,
       'max_supply': 21000000.0,
       'quotes': {'USD': {'price': 307.839284363,
         'volume_24h': 1364820049.07966,
         'market_cap': 5538802172.0,
         'percent_change_1h': 0.61,
         'percent_change_24h': 1.64,
         'percent_change_7d': 9.14}},
       'last_updated': 1567986967},
      '2': {'id': 2,
       'name': 'Litecoin',
       'symbol': 'LTC',
       'website_slug': 'litecoin',
       'rank': 5,
       'circulating_supply': 63212442.0,
       'total_supply': 63212442.0,
       'max_supply': 84000000.0,
       'quotes': {'USD': {'price': 70.6729059892,
         'volume_24h': 2779409088.79373,
         'market_cap': 4467406945.0,
         'percent_change_1h': 0.05,
         'percent_change_24h': 2.12,
         'percent_change_7d': 7.0}},
       'last_updated': 1567986966},
      '825': {'id': 825,
       'name': 'Tether',
       'symbol': 'USDT',
       'website_slug': 'tether',
       'rank': 6,
       'circulating_supply': 4071193568.0,
       'total_supply': 4160057493.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0041729464,
         'volume_24h': 16251849805.0479,
         'market_cap': 4088182441.0,
         'percent_change_1h': -0.13,
         'percent_change_24h': -0.07,
         'percent_change_7d': 0.13}},
       'last_updated': 1567986979},
      '1765': {'id': 1765,
       'name': 'EOS',
       'symbol': 'EOS',
       'website_slug': 'eos',
       'rank': 7,
       'circulating_supply': 930917384.0,
       'total_supply': 1027617396.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 3.7771337806,
         'volume_24h': 2108333082.59896,
         'market_cap': 3516199499.0,
         'percent_change_1h': 1.01,
         'percent_change_24h': 5.41,
         'percent_change_7d': 16.04}},
       'last_updated': 1567986966},
      '1839': {'id': 1839,
       'name': 'Binance Coin',
       'symbol': 'BNB',
       'website_slug': 'binance-coin',
       'rank': 8,
       'circulating_supply': 155536713.0,
       'total_supply': 187536713.0,
       'max_supply': 187536713.0,
       'quotes': {'USD': {'price': 22.5544842169,
         'volume_24h': 174945393.476032,
         'market_cap': 3508050339.0,
         'percent_change_1h': -0.04,
         'percent_change_24h': -0.25,
         'percent_change_7d': 4.81}},
       'last_updated': 1567986965},
      '3602': {'id': 3602,
       'name': 'Bitcoin SV',
       'symbol': 'BSV',
       'website_slug': 'bitcoin-sv',
       'rank': 9,
       'circulating_supply': 17854986.0,
       'total_supply': 17854986.0,
       'max_supply': 21000000.0,
       'quotes': {'USD': {'price': 135.460960018,
         'volume_24h': 306987238.114515,
         'market_cap': 2418653511.0,
         'percent_change_1h': 0.76,
         'percent_change_24h': 0.24,
         'percent_change_7d': 4.52}},
       'last_updated': 1567986970},
      '328': {'id': 328,
       'name': 'Monero',
       'symbol': 'XMR',
       'website_slug': 'monero',
       'rank': 10,
       'circulating_supply': 17198377.0,
       'total_supply': 17198377.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 77.4009647372,
         'volume_24h': 58196952.2596086,
         'market_cap': 1331170974.0,
         'percent_change_1h': 0.04,
         'percent_change_24h': -1.37,
         'percent_change_7d': 9.72}},
       'last_updated': 1567986962},
      '2010': {'id': 2010,
       'name': 'Cardano',
       'symbol': 'ADA',
       'website_slug': 'cardano',
       'rank': 11,
       'circulating_supply': 25927070538.0,
       'total_supply': 31112483745.0,
       'max_supply': 45000000000.0,
       'quotes': {'USD': {'price': 0.0467211924,
         'volume_24h': 46921959.0341445,
         'market_cap': 1211343650.0,
         'percent_change_1h': -0.19,
         'percent_change_24h': 0.97,
         'percent_change_7d': 5.62}},
       'last_updated': 1567986963},
      '512': {'id': 512,
       'name': 'Stellar',
       'symbol': 'XLM',
       'website_slug': 'stellar',
       'rank': 12,
       'circulating_supply': 19747411152.0,
       'total_supply': 105303236854.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0608573316,
         'volume_24h': 162875839.151028,
         'market_cap': 1201774748.0,
         'percent_change_1h': -0.61,
         'percent_change_24h': 2.26,
         'percent_change_7d': -2.37}},
       'last_updated': 1567986963},
      '3957': {'id': 3957,
       'name': 'UNUS SED LEO',
       'symbol': 'LEO',
       'website_slug': 'unus-sed-leo',
       'rank': 13,
       'circulating_supply': 999498893.0,
       'total_supply': 999498893.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0702280156,
         'volume_24h': 5198611.21074775,
         'market_cap': 1069691717.0,
         'percent_change_1h': 0.03,
         'percent_change_24h': 0.53,
         'percent_change_7d': -6.24}},
       'last_updated': 1567986969},
      '1958': {'id': 1958,
       'name': 'TRON',
       'symbol': 'TRX',
       'website_slug': 'tron',
       'rank': 14,
       'circulating_supply': 66682072191.0,
       'total_supply': 99281283754.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.015836497,
         'volume_24h': 564814767.206134,
         'market_cap': 1056010437.0,
         'percent_change_1h': 0.31,
         'percent_change_24h': 2.16,
         'percent_change_7d': 2.14}},
       'last_updated': 1567986964},
      '2502': {'id': 2502,
       'name': 'Huobi Token',
       'symbol': 'HT',
       'website_slug': 'huobi-token',
       'rank': 15,
       'circulating_supply': 245880576.0,
       'total_supply': 500000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 4.0502712371,
         'volume_24h': 68378783.9119013,
         'market_cap': 995883025.0,
         'percent_change_1h': -0.04,
         'percent_change_24h': 0.3,
         'percent_change_7d': 6.75}},
       'last_updated': 1567986965},
      '131': {'id': 131,
       'name': 'Dash',
       'symbol': 'DASH',
       'website_slug': 'dash',
       'rank': 16,
       'circulating_supply': 9028603.0,
       'total_supply': 9028603.0,
       'max_supply': 18900000.0,
       'quotes': {'USD': {'price': 87.7598564988,
         'volume_24h': 170640485.580917,
         'market_cap': 792348888.0,
         'percent_change_1h': 0.25,
         'percent_change_24h': 3.7,
         'percent_change_7d': 9.83}},
       'last_updated': 1567986962},
      '1321': {'id': 1321,
       'name': 'Ethereum Classic',
       'symbol': 'ETC',
       'website_slug': 'ethereum-classic',
       'rank': 17,
       'circulating_supply': 113321367.0,
       'total_supply': 113321367.0,
       'max_supply': 210000000.0,
       'quotes': {'USD': {'price': 6.6651859368,
         'volume_24h': 510780435.914381,
         'market_cap': 755307982.0,
         'percent_change_1h': 0.08,
         'percent_change_24h': -0.4,
         'percent_change_7d': 6.51}},
       'last_updated': 1567986964},
      '2011': {'id': 2011,
       'name': 'Tezos',
       'symbol': 'XTZ',
       'website_slug': 'tezos',
       'rank': 18,
       'circulating_supply': 660373612.0,
       'total_supply': 801312599.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.065791645,
         'volume_24h': 10249468.0514868,
         'market_cap': 703820678.0,
         'percent_change_1h': 0.4,
         'percent_change_24h': 3.63,
         'percent_change_7d': 3.37}},
       'last_updated': 1567986963},
      '1720': {'id': 1720,
       'name': 'IOTA',
       'symbol': 'MIOTA',
       'website_slug': 'iota',
       'rank': 19,
       'circulating_supply': 2779530283.0,
       'total_supply': 2779530283.0,
       'max_supply': 2779530283.0,
       'quotes': {'USD': {'price': 0.244935781,
         'volume_24h': 3992432.69753796,
         'market_cap': 680806421.0,
         'percent_change_1h': 0.18,
         'percent_change_24h': 1.09,
         'percent_change_7d': 1.15}},
       'last_updated': 1567986962},
      '1376': {'id': 1376,
       'name': 'NEO',
       'symbol': 'NEO',
       'website_slug': 'neo',
       'rank': 20,
       'circulating_supply': 70538831.0,
       'total_supply': 100000000.0,
       'max_supply': 100000000.0,
       'quotes': {'USD': {'price': 9.3080541577,
         'volume_24h': 242130212.284564,
         'market_cap': 656579259.0,
         'percent_change_1h': -0.02,
         'percent_change_24h': 1.88,
         'percent_change_7d': 5.97}},
       'last_updated': 1567986963},
      '1975': {'id': 1975,
       'name': 'Chainlink',
       'symbol': 'LINK',
       'website_slug': 'chainlink',
       'rank': 21,
       'circulating_supply': 350000000.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.8350587475,
         'volume_24h': 88711837.6655027,
         'market_cap': 642270562.0,
         'percent_change_1h': -0.94,
         'percent_change_24h': 3.05,
         'percent_change_7d': 4.24}},
       'last_updated': 1567986963},
      '3794': {'id': 3794,
       'name': 'Cosmos',
       'symbol': 'ATOM',
       'website_slug': 'cosmos',
       'rank': 22,
       'circulating_supply': 190688439.0,
       'total_supply': 237928231.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 2.7301060196,
         'volume_24h': 195984085.202543,
         'market_cap': 520599656.0,
         'percent_change_1h': 1.48,
         'percent_change_24h': 24.2,
         'percent_change_7d': 28.18}},
       'last_updated': 1567986970},
      '1518': {'id': 1518,
       'name': 'Maker',
       'symbol': 'MKR',
       'website_slug': 'maker',
       'rank': 23,
       'circulating_supply': 1000000.0,
       'total_supply': 1000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 448.311031725,
         'volume_24h': 12828099.8792397,
         'market_cap': 448311032.0,
         'percent_change_1h': -0.25,
         'percent_change_24h': 2.31,
         'percent_change_7d': -4.06}},
       'last_updated': 1567986962},
      '3408': {'id': 3408,
       'name': 'USD Coin',
       'symbol': 'USDC',
       'website_slug': 'usd-coin',
       'rank': 24,
       'circulating_supply': 442376632.0,
       'total_supply': 443564278.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0028530363,
         'volume_24h': 183802534.634433,
         'market_cap': 443638749.0,
         'percent_change_1h': -0.07,
         'percent_change_24h': 0.14,
         'percent_change_7d': 0.19}},
       'last_updated': 1567986968},
      '873': {'id': 873,
       'name': 'NEM',
       'symbol': 'XEM',
       'website_slug': 'nem',
       'rank': 25,
       'circulating_supply': 8999999999.0,
       'total_supply': 8999999999.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0471053699,
         'volume_24h': 14639606.5271335,
         'market_cap': 423948329.0,
         'percent_change_1h': -0.1,
         'percent_change_24h': 0.28,
         'percent_change_7d': -3.86}},
       'last_updated': 1567986962},
      '2566': {'id': 2566,
       'name': 'Ontology',
       'symbol': 'ONT',
       'website_slug': 'ontology',
       'rank': 26,
       'circulating_supply': 533383967.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.7602078571,
         'volume_24h': 80502074.9715123,
         'market_cap': 405482683.0,
         'percent_change_1h': 0.54,
         'percent_change_24h': 5.96,
         'percent_change_7d': 6.99}},
       'last_updated': 1567986966},
      '3635': {'id': 3635,
       'name': 'Crypto.com Chain',
       'symbol': 'CRO',
       'website_slug': 'crypto-com-chain',
       'rank': 27,
       'circulating_supply': 9664383562.0,
       'total_supply': 100000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0396923913,
         'volume_24h': 9998074.81454569,
         'market_cap': 383602494.0,
         'percent_change_1h': 0.2,
         'percent_change_24h': -3.82,
         'percent_change_7d': 1.62}},
       'last_updated': 1567986968},
      '3085': {'id': 3085,
       'name': 'INO COIN',
       'symbol': 'INO',
       'website_slug': 'ino-coin',
       'rank': 28,
       'circulating_supply': 180003180.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 2.0415480918,
         'volume_24h': 16043.3667766244,
         'market_cap': 367485149.0,
         'percent_change_1h': -0.05,
         'percent_change_24h': -0.88,
         'percent_change_7d': 7.94}},
       'last_updated': 1567986968},
      '1437': {'id': 1437,
       'name': 'Zcash',
       'symbol': 'ZEC',
       'website_slug': 'zcash',
       'rank': 29,
       'circulating_supply': 7371831.0,
       'total_supply': 7371831.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 48.6357708664,
         'volume_24h': 136211482.207761,
         'market_cap': 358534696.0,
         'percent_change_1h': 0.35,
         'percent_change_24h': 3.02,
         'percent_change_7d': 7.39}},
       'last_updated': 1567986963},
      '74': {'id': 74,
       'name': 'Dogecoin',
       'symbol': 'DOGE',
       'website_slug': 'dogecoin',
       'rank': 30,
       'circulating_supply': 121120350283.0,
       'total_supply': 121120350283.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0025085651,
         'volume_24h': 31599406.2302629,
         'market_cap': 303838282.0,
         'percent_change_1h': 0.44,
         'percent_change_24h': -0.48,
         'percent_change_7d': 0.86}},
       'last_updated': 1567986963},
      '3704': {'id': 3704,
       'name': 'V Systems',
       'symbol': 'VSYS',
       'website_slug': 'v-systems',
       'rank': 31,
       'circulating_supply': 1799564030.0,
       'total_supply': 3704422494.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1529149234,
         'volume_24h': 6440082.94647771,
         'market_cap': 275180196.0,
         'percent_change_1h': 2.58,
         'percent_change_24h': 1.34,
         'percent_change_7d': 8.85}},
       'last_updated': 1567986969},
      '3662': {'id': 3662,
       'name': 'HedgeTrade',
       'symbol': 'HEDG',
       'website_slug': 'hedgetrade',
       'rank': 32,
       'circulating_supply': 288393355.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.9386148943,
         'volume_24h': 562481.290855677,
         'market_cap': 270690298.0,
         'percent_change_1h': 1.09,
         'percent_change_24h': 0.68,
         'percent_change_7d': 7.8}},
       'last_updated': 1567986969},
      '1168': {'id': 1168,
       'name': 'Decred',
       'symbol': 'DCR',
       'website_slug': 'decred',
       'rank': 33,
       'circulating_supply': 10357002.0,
       'total_supply': 10357002.0,
       'max_supply': 21000000.0,
       'quotes': {'USD': {'price': 24.9245061688,
         'volume_24h': 9153396.02954392,
         'market_cap': 258143153.0,
         'percent_change_1h': 0.24,
         'percent_change_24h': 1.7,
         'percent_change_7d': 4.49}},
       'last_updated': 1567986962},
      '3330': {'id': 3330,
       'name': 'Paxos Standard Token',
       'symbol': 'PAX',
       'website_slug': 'paxos-standard-token',
       'rank': 34,
       'circulating_supply': 237557039.0,
       'total_supply': 237650479.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0027367707,
         'volume_24h': 333583711.51625,
         'market_cap': 238207178.0,
         'percent_change_1h': -0.15,
         'percent_change_24h': 0.1,
         'percent_change_7d': 0.08}},
       'last_updated': 1567986968},
      '1697': {'id': 1697,
       'name': 'Basic Attention Token',
       'symbol': 'BAT',
       'website_slug': 'basic-attention-token',
       'rank': 35,
       'circulating_supply': 1329725522.0,
       'total_supply': 1500000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1763125219,
         'volume_24h': 20683588.5762913,
         'market_cap': 234447260.0,
         'percent_change_1h': -0.01,
         'percent_change_24h': 0.96,
         'percent_change_7d': -2.04}},
       'last_updated': 1567986963},
      '3916': {'id': 3916,
       'name': 'ThoreNext',
       'symbol': 'THX',
       'website_slug': 'thorenext',
       'rank': 36,
       'circulating_supply': 21652254.0,
       'total_supply': 210000000.0,
       'max_supply': 210000000.0,
       'quotes': {'USD': {'price': 10.3291522073,
         'volume_24h': 171382.286789324,
         'market_cap': 223649427.0,
         'percent_change_1h': -0.0,
         'percent_change_24h': -0.76,
         'percent_change_7d': 13.54}},
       'last_updated': 1567986970},
      '3077': {'id': 3077,
       'name': 'VeChain',
       'symbol': 'VET',
       'website_slug': 'vechain',
       'rank': 37,
       'circulating_supply': 55454734800.0,
       'total_supply': 86712634466.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0040010012,
         'volume_24h': 28491347.4353131,
         'market_cap': 221874462.0,
         'percent_change_1h': -0.37,
         'percent_change_24h': 3.33,
         'percent_change_7d': -2.65}},
       'last_updated': 1567986968},
      '1684': {'id': 1684,
       'name': 'Qtum',
       'symbol': 'QTUM',
       'website_slug': 'qtum',
       'rank': 38,
       'circulating_supply': 96003140.0,
       'total_supply': 101753160.0,
       'max_supply': 107822406.0,
       'quotes': {'USD': {'price': 2.1285414505,
         'volume_24h': 172749046.05945,
         'market_cap': 204346663.0,
         'percent_change_1h': 0.28,
         'percent_change_24h': 4.64,
         'percent_change_7d': 1.52}},
       'last_updated': 1567986964},
      '2563': {'id': 2563,
       'name': 'TrueUSD',
       'symbol': 'TUSD',
       'website_slug': 'trueusd',
       'rank': 39,
       'circulating_supply': 198868513.0,
       'total_supply': 198868513.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.002233539,
         'volume_24h': 297530902.202327,
         'market_cap': 199312694.0,
         'percent_change_1h': -0.11,
         'percent_change_24h': -0.02,
         'percent_change_7d': 0.14}},
       'last_updated': 1567986967},
      '2083': {'id': 2083,
       'name': 'Bitcoin Gold',
       'symbol': 'BTG',
       'website_slug': 'bitcoin-gold',
       'rank': 40,
       'circulating_supply': 17513924.0,
       'total_supply': 17513924.0,
       'max_supply': 21000000.0,
       'quotes': {'USD': {'price': 10.7262920328,
         'volume_24h': 12800992.0387199,
         'market_cap': 187859459.0,
         'percent_change_1h': 0.34,
         'percent_change_24h': 0.38,
         'percent_change_7d': 1.19}},
       'last_updated': 1567986964},
      '3144': {'id': 3144,
       'name': 'ThoreCoin',
       'symbol': 'THR',
       'website_slug': 'thorecoin',
       'rank': 41,
       'circulating_supply': 86686.0,
       'total_supply': 100000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1877.64792929,
         'volume_24h': 181405.962563868,
         'market_cap': 162765829.0,
         'percent_change_1h': -0.0,
         'percent_change_24h': -0.73,
         'percent_change_7d': 7.51}},
       'last_updated': 1567986966},
      '3351': {'id': 3351,
       'name': 'ZB',
       'symbol': 'ZB',
       'website_slug': 'zb',
       'rank': 42,
       'circulating_supply': 463288810.0,
       'total_supply': 2100000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.3394690512,
         'volume_24h': 75920593.7250561,
         'market_cap': 157272213.0,
         'percent_change_1h': 0.02,
         'percent_change_24h': 0.52,
         'percent_change_7d': 3.91}},
       'last_updated': 1567986967},
      '1808': {'id': 1808,
       'name': 'OmiseGO',
       'symbol': 'OMG',
       'website_slug': 'omisego',
       'rank': 43,
       'circulating_supply': 140245398.0,
       'total_supply': 140245398.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0902357081,
         'volume_24h': 41778303.7348833,
         'market_cap': 152900541.0,
         'percent_change_1h': -0.18,
         'percent_change_24h': 1.84,
         'percent_change_7d': 0.05}},
       'last_updated': 1567986964},
      '2907': {'id': 2907,
       'name': 'Karatgold Coin',
       'symbol': 'KBC',
       'website_slug': 'karatgold-coin',
       'rank': 44,
       'circulating_supply': 4042622937.0,
       'total_supply': 12000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0348225577,
         'volume_24h': 4452194.08701351,
         'market_cap': 140774470.0,
         'percent_change_1h': 2.07,
         'percent_change_24h': 49.33,
         'percent_change_7d': 78.21}},
       'last_updated': 1567986968},
      '2577': {'id': 2577,
       'name': 'Ravencoin',
       'symbol': 'RVN',
       'website_slug': 'ravencoin',
       'rank': 45,
       'circulating_supply': 4376415000.0,
       'total_supply': 4376415000.0,
       'max_supply': 21000000000.0,
       'quotes': {'USD': {'price': 0.0319696639,
         'volume_24h': 10743360.4854761,
         'market_cap': 139912517.0,
         'percent_change_1h': -0.51,
         'percent_change_24h': -1.59,
         'percent_change_7d': 2.62}},
       'last_updated': 1567986965},
      '2087': {'id': 2087,
       'name': 'KuCoin Shares',
       'symbol': 'KCS',
       'website_slug': 'kucoin-shares',
       'rank': 46,
       'circulating_supply': 88086720.0,
       'total_supply': 178086720.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.5266510516,
         'volume_24h': 18182355.1072269,
         'market_cap': 134477684.0,
         'percent_change_1h': -1.03,
         'percent_change_24h': 3.64,
         'percent_change_7d': 5.2}},
       'last_updated': 1567986963},
      '1214': {'id': 1214,
       'name': 'Lisk',
       'symbol': 'LSK',
       'website_slug': 'lisk',
       'rank': 47,
       'circulating_supply': 119969711.0,
       'total_supply': 135066124.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0615349625,
         'volume_24h': 1663407.47767765,
         'market_cap': 127352043.0,
         'percent_change_1h': -0.42,
         'percent_change_24h': -1.35,
         'percent_change_7d': -4.76}},
       'last_updated': 1567986962},
      '1567': {'id': 1567,
       'name': 'Nano',
       'symbol': 'NANO',
       'website_slug': 'nano',
       'rank': 48,
       'circulating_supply': 133248297.0,
       'total_supply': 133248297.0,
       'max_supply': 133248297.0,
       'quotes': {'USD': {'price': 0.9248234494,
         'volume_24h': 2524274.72513595,
         'market_cap': 123231150.0,
         'percent_change_1h': 0.11,
         'percent_change_24h': -0.71,
         'percent_change_7d': -3.71}},
       'last_updated': 1567986962},
      '3718': {'id': 3718,
       'name': 'BitTorrent',
       'symbol': 'BTT',
       'website_slug': 'bittorrent',
       'rank': 49,
       'circulating_supply': 212116500000.0,
       'total_supply': 990000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0005730135,
         'volume_24h': 92677890.390371,
         'market_cap': 121545608.0,
         'percent_change_1h': -0.35,
         'percent_change_24h': 8.48,
         'percent_change_7d': 9.46}},
       'last_updated': 1567986971},
      '3788': {'id': 3788,
       'name': 'NEXT',
       'symbol': 'NET',
       'website_slug': 'next',
       'rank': 50,
       'circulating_supply': 50269268.0,
       'total_supply': 973628555.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 2.4166392974,
         'volume_24h': 5000624.74062421,
         'market_cap': 121482689.0,
         'percent_change_1h': -2.93,
         'percent_change_24h': 13.78,
         'percent_change_7d': 53.28}},
       'last_updated': 1567986970},
      '1104': {'id': 1104,
       'name': 'Augur',
       'symbol': 'REP',
       'website_slug': 'augur',
       'rank': 51,
       'circulating_supply': 11000000.0,
       'total_supply': 11000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 11.0404470596,
         'volume_24h': 9862553.3646413,
         'market_cap': 121444918.0,
         'percent_change_1h': 0.49,
         'percent_change_24h': 0.11,
         'percent_change_7d': 37.33}},
       'last_updated': 1567986962},
      '4030': {'id': 4030,
       'name': 'Algorand',
       'symbol': 'ALGO',
       'website_slug': 'algorand',
       'rank': 52,
       'circulating_supply': 311885637.0,
       'total_supply': 2843157480.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.3812074399,
         'volume_24h': 50638307.1299149,
         'market_cap': 118893125.0,
         'percent_change_1h': 4.04,
         'percent_change_24h': 1.66,
         'percent_change_7d': -2.11}},
       'last_updated': 1567986970},
      '2897': {'id': 2897,
       'name': 'Clipper Coin',
       'symbol': 'CCCX',
       'website_slug': 'clipper-coin',
       'rank': 53,
       'circulating_supply': 3780570996.0,
       'total_supply': 5000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0309145682,
         'volume_24h': 36853.3044676971,
         'market_cap': 116874720.0,
         'percent_change_1h': 0.31,
         'percent_change_24h': -2.97,
         'percent_change_7d': 23.48}},
       'last_updated': 1567986965},
      '3116': {'id': 3116,
       'name': 'Insight Chain',
       'symbol': 'INB',
       'website_slug': 'insight-chain',
       'rank': 54,
       'circulating_supply': 349902689.0,
       'total_supply': 10000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.3290289668,
         'volume_24h': 9964643.96392558,
         'market_cap': 115128120.0,
         'percent_change_1h': -0.36,
         'percent_change_24h': 1.61,
         'percent_change_7d': -3.03}},
       'last_updated': 1567986966},
      '2222': {'id': 2222,
       'name': 'Bitcoin Diamond',
       'symbol': 'BCD',
       'website_slug': 'bitcoin-diamond',
       'rank': 55,
       'circulating_supply': 186492898.0,
       'total_supply': 189492898.0,
       'max_supply': 210000000.0,
       'quotes': {'USD': {'price': 0.6046094925,
         'volume_24h': 3351342.11034825,
         'market_cap': 112755376.0,
         'percent_change_1h': 0.18,
         'percent_change_24h': -2.54,
         'percent_change_7d': -6.07}},
       'last_updated': 1567986964},
      '4203': {'id': 4203,
       'name': 'Oasis City',
       'symbol': 'OSC',
       'website_slug': 'oasis-city',
       'rank': 56,
       'circulating_supply': 2119178872.0,
       'total_supply': 12000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0529017516,
         'volume_24h': 800805.418820122,
         'market_cap': 112108274.0,
         'percent_change_1h': 1.55,
         'percent_change_24h': -0.56,
         'percent_change_7d': 12.85}},
       'last_updated': 1567986970},
      '2453': {'id': 2453,
       'name': 'EDUCare',
       'symbol': 'EKT',
       'website_slug': 'educare',
       'rank': 57,
       'circulating_supply': 750000000.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1475238797,
         'volume_24h': 6749082.09815311,
         'market_cap': 110642910.0,
         'percent_change_1h': -0.14,
         'percent_change_24h': -2.74,
         'percent_change_7d': 3.59}},
       'last_updated': 1567986964},
      '2682': {'id': 2682,
       'name': 'Holo',
       'symbol': 'HOT',
       'website_slug': 'holo',
       'rank': 58,
       'circulating_supply': 133214575156.0,
       'total_supply': 177619433541.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0008172918,
         'volume_24h': 5659074.37495841,
         'market_cap': 108875185.0,
         'percent_change_1h': 0.22,
         'percent_change_24h': 1.49,
         'percent_change_7d': 2.8}},
       'last_updated': 1567986966},
      '1274': {'id': 1274,
       'name': 'Waves',
       'symbol': 'WAVES',
       'website_slug': 'waves',
       'rank': 59,
       'circulating_supply': 100000000.0,
       'total_supply': 100000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0654831657,
         'volume_24h': 6955279.43032668,
         'market_cap': 106548317.0,
         'percent_change_1h': -0.24,
         'percent_change_24h': -1.19,
         'percent_change_7d': -4.0}},
       'last_updated': 1567986962},
      '2416': {'id': 2416,
       'name': 'THETA',
       'symbol': 'THETA',
       'website_slug': 'theta',
       'rank': 60,
       'circulating_supply': 870502690.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1195411408,
         'volume_24h': 625419.083708336,
         'market_cap': 104060885.0,
         'percent_change_1h': -0.11,
         'percent_change_24h': -0.57,
         'percent_change_7d': 2.31}},
       'last_updated': 1567986965},
      '109': {'id': 109,
       'name': 'DigiByte',
       'symbol': 'DGB',
       'website_slug': 'digibyte',
       'rank': 61,
       'circulating_supply': 12241913044.0,
       'total_supply': 12241913044.0,
       'max_supply': 21000000000.0,
       'quotes': {'USD': {'price': 0.008478072,
         'volume_24h': 3277372.4960745,
         'market_cap': 103787820.0,
         'percent_change_1h': 0.34,
         'percent_change_24h': 6.11,
         'percent_change_7d': 2.41}},
       'last_updated': 1567986961},
      '2349': {'id': 2349,
       'name': 'Mixin',
       'symbol': 'XIN',
       'website_slug': 'mixin',
       'rank': 62,
       'circulating_supply': 458732.0,
       'total_supply': 1000000.0,
       'max_supply': 1000000.0,
       'quotes': {'USD': {'price': 220.427009275,
         'volume_24h': 18027913.0137782,
         'market_cap': 101116846.0,
         'percent_change_1h': 0.55,
         'percent_change_24h': -1.85,
         'percent_change_7d': -1.75}},
       'last_updated': 1567986965},
      '2099': {'id': 2099,
       'name': 'ICON',
       'symbol': 'ICX',
       'website_slug': 'icon',
       'rank': 63,
       'circulating_supply': 492837269.0,
       'total_supply': 800460000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.2017371511,
         'volume_24h': 10703516.9304418,
         'market_cap': 99423587.0,
         'percent_change_1h': -0.12,
         'percent_change_24h': -0.06,
         'percent_change_7d': -5.23}},
       'last_updated': 1567986964},
      '1903': {'id': 1903,
       'name': 'HyperCash',
       'symbol': 'HC',
       'website_slug': 'hypercash',
       'rank': 64,
       'circulating_supply': 43529781.0,
       'total_supply': 43529781.0,
       'max_supply': 84000000.0,
       'quotes': {'USD': {'price': 2.2753272817,
         'volume_24h': 12021498.4844898,
         'market_cap': 99044498.0,
         'percent_change_1h': 0.76,
         'percent_change_24h': 9.39,
         'percent_change_7d': 6.79}},
       'last_updated': 1567986963},
      '1896': {'id': 1896,
       'name': '0x',
       'symbol': 'ZRX',
       'website_slug': '0x',
       'rank': 65,
       'circulating_supply': 600475853.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1618180068,
         'volume_24h': 7078877.94994805,
         'market_cap': 97167806.0,
         'percent_change_1h': -0.26,
         'percent_change_24h': -1.47,
         'percent_change_7d': 1.19}},
       'last_updated': 1567986965},
      '372': {'id': 372,
       'name': 'Bytecoin',
       'symbol': 'BCN',
       'website_slug': 'bytecoin-bcn',
       'rank': 66,
       'circulating_supply': 184066828814.0,
       'total_supply': 184066828814.0,
       'max_supply': 184470000000.0,
       'quotes': {'USD': {'price': 0.0005066692,
         'volume_24h': 14970.7968711882,
         'market_cap': 93260991.0,
         'percent_change_1h': 12.69,
         'percent_change_24h': 3.83,
         'percent_change_7d': -10.02}},
       'last_updated': 1567986961},
      '291': {'id': 291,
       'name': 'MaidSafeCoin',
       'symbol': 'MAID',
       'website_slug': 'maidsafecoin',
       'rank': 67,
       'circulating_supply': 452552412.0,
       'total_supply': 452552412.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.2030835173,
         'volume_24h': 288155.653142957,
         'market_cap': 91905936.0,
         'percent_change_1h': -0.09,
         'percent_change_24h': 6.1,
         'percent_change_7d': 20.74}},
       'last_updated': 1567986961},
      '3657': {'id': 3657,
       'name': 'Lambda',
       'symbol': 'LAMB',
       'website_slug': 'lambda',
       'rank': 68,
       'circulating_supply': 629686531.0,
       'total_supply': 6000000000.0,
       'max_supply': 10000000000.0,
       'quotes': {'USD': {'price': 0.1454219419,
         'volume_24h': 51421106.1402782,
         'market_cap': 91570238.0,
         'percent_change_1h': -0.89,
         'percent_change_24h': 0.59,
         'percent_change_7d': -13.19}},
       'last_updated': 1567986969},
      '463': {'id': 463,
       'name': 'BitShares',
       'symbol': 'BTS',
       'website_slug': 'bitshares',
       'rank': 69,
       'circulating_supply': 2741570000.0,
       'total_supply': 2741570000.0,
       'max_supply': 3600570502.0,
       'quotes': {'USD': {'price': 0.0328302036,
         'volume_24h': 2056071.75465928,
         'market_cap': 90006301.0,
         'percent_change_1h': -0.09,
         'percent_change_24h': 0.09,
         'percent_change_7d': -0.06}},
       'last_updated': 1567986961},
      '2874': {'id': 2874,
       'name': 'Aurora',
       'symbol': 'AOA',
       'website_slug': 'aurora',
       'rank': 70,
       'circulating_supply': 6542330148.0,
       'total_supply': 10000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0135382054,
         'volume_24h': 3304096.99449387,
         'market_cap': 88571409.0,
         'percent_change_1h': 4.73,
         'percent_change_24h': 8.4,
         'percent_change_7d': -1.17}},
       'last_updated': 1567986966},
      '2603': {'id': 2603,
       'name': 'Pundi X',
       'symbol': 'NPXS',
       'website_slug': 'pundi-x',
       'rank': 71,
       'circulating_supply': 235621468515.0,
       'total_supply': 261834927333.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0003698659,
         'volume_24h': 2143007.86120292,
         'market_cap': 87148338.0,
         'percent_change_1h': -0.35,
         'percent_change_24h': 1.17,
         'percent_change_7d': -4.64}},
       'last_updated': 1567986966},
      '1521': {'id': 1521,
       'name': 'Komodo',
       'symbol': 'KMD',
       'website_slug': 'komodo',
       'rank': 72,
       'circulating_supply': 115832288.0,
       'total_supply': 115832288.0,
       'max_supply': 200000000.0,
       'quotes': {'USD': {'price': 0.7383913045,
         'volume_24h': 2559416.55496776,
         'market_cap': 85529555.0,
         'percent_change_1h': 0.29,
         'percent_change_24h': -2.96,
         'percent_change_7d': 13.1}},
       'last_updated': 1567986962},
      '3134': {'id': 3134,
       'name': 'ETERNAL TOKEN',
       'symbol': 'XET',
       'website_slug': 'eternal-token',
       'rank': 73,
       'circulating_supply': 93470000.0,
       'total_supply': 200000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.8993041338,
         'volume_24h': 102638.034365879,
         'market_cap': 84057957.0,
         'percent_change_1h': 0.08,
         'percent_change_24h': -0.18,
         'percent_change_7d': 16.36}},
       'last_updated': 1567986970},
      '2405': {'id': 2405,
       'name': 'IOST',
       'symbol': 'IOST',
       'website_slug': 'iostoken',
       'rank': 74,
       'circulating_supply': 12013965609.0,
       'total_supply': 21000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0069663965,
         'volume_24h': 14234491.3196043,
         'market_cap': 83694047.0,
         'percent_change_1h': 0.15,
         'percent_change_24h': -0.69,
         'percent_change_7d': -0.29}},
       'last_updated': 1567986966},
      '3829': {'id': 3829,
       'name': 'Nash Exchange',
       'symbol': 'NEX',
       'website_slug': 'nash-exchange',
       'rank': 75,
       'circulating_supply': 36196678.0,
       'total_supply': 56296100.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 2.2429477054,
         'volume_24h': 3452852.16449743,
         'market_cap': 81187256.0,
         'percent_change_1h': 0.53,
         'percent_change_24h': 1.8,
         'percent_change_7d': -2.13}},
       'last_updated': 1567986971},
      '1866': {'id': 1866,
       'name': 'Bytom',
       'symbol': 'BTM',
       'website_slug': 'bytom',
       'rank': 76,
       'circulating_supply': 1002499275.0,
       'total_supply': 1407000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0802018006,
         'volume_24h': 6422791.56381057,
         'market_cap': 80402247.0,
         'percent_change_1h': -0.45,
         'percent_change_24h': 3.65,
         'percent_change_7d': 2.61}},
       'last_updated': 1567986963},
      '3840': {'id': 3840,
       'name': '1irstcoin',
       'symbol': 'FST',
       'website_slug': '1irstcoin',
       'rank': 77,
       'circulating_supply': 22085000.0,
       'total_supply': 100000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 3.5771944604,
         'volume_24h': 207978.543237787,
         'market_cap': 79002340.0,
         'percent_change_1h': 0.8,
         'percent_change_24h': -3.8,
         'percent_change_7d': 3.65}},
       'last_updated': 1567986970},
      '3218': {'id': 3218,
       'name': 'Energi',
       'symbol': 'NRG',
       'website_slug': 'energi',
       'rank': 78,
       'circulating_supply': 20340654.0,
       'total_supply': 20340654.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 3.862153941,
         'volume_24h': 516183.106836966,
         'market_cap': 78558735.0,
         'percent_change_1h': -0.14,
         'percent_change_24h': -4.56,
         'percent_change_7d': -23.91}},
       'last_updated': 1567986967},
      '2308': {'id': 2308,
       'name': 'Dai',
       'symbol': 'DAI',
       'website_slug': 'dai',
       'rank': 79,
       'circulating_supply': 77905054.0,
       'total_supply': 77905054.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.0053430587,
         'volume_24h': 24715418.7812616,
         'market_cap': 78321305.0,
         'percent_change_1h': -0.48,
         'percent_change_24h': 0.31,
         'percent_change_7d': 0.1}},
       'last_updated': 1567986964},
      '4049': {'id': 4049,
       'name': 'Bitbook Gambling',
       'symbol': 'BXK',
       'website_slug': 'bitbook-gambling',
       'rank': 80,
       'circulating_supply': 368387491.0,
       'total_supply': 741456054.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.2092264726,
         'volume_24h': 916748.96326273,
         'market_cap': 77076415.0,
         'percent_change_1h': -1.27,
         'percent_change_24h': -2.07,
         'percent_change_7d': 2.58}},
       'last_updated': 1567986972},
      '1042': {'id': 1042,
       'name': 'Siacoin',
       'symbol': 'SC',
       'website_slug': 'siacoin',
       'rank': 81,
       'circulating_supply': 41817047634.0,
       'total_supply': 41817047634.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0018405348,
         'volume_24h': 1226444.75796823,
         'market_cap': 76965730.0,
         'percent_change_1h': -2.79,
         'percent_change_24h': -0.43,
         'percent_change_7d': 0.81}},
       'last_updated': 1567986962},
      '213': {'id': 213,
       'name': 'MonaCoin',
       'symbol': 'MONA',
       'website_slug': 'monacoin',
       'rank': 82,
       'circulating_supply': 65729675.0,
       'total_supply': 65729675.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 1.1246887263,
         'volume_24h': 1375851.46167092,
         'market_cap': 73925424.0,
         'percent_change_1h': -0.11,
         'percent_change_24h': -0.06,
         'percent_change_7d': -10.46}},
       'last_updated': 1567986961},
      '3812': {'id': 3812,
       'name': 'Flexacoin',
       'symbol': 'FXC',
       'website_slug': 'flexacoin',
       'rank': 83,
       'circulating_supply': 20586445749.0,
       'total_supply': 20586445749.0,
       'max_supply': 100000000000.0,
       'quotes': {'USD': {'price': 0.0034560374,
         'volume_24h': 20713.1329864326,
         'market_cap': 71147527.0,
         'percent_change_1h': 2.08,
         'percent_change_24h': 18.16,
         'percent_change_7d': 33.04}},
       'last_updated': 1567986969},
      '3987': {'id': 3987,
       'name': 'Beldex',
       'symbol': 'BDX',
       'website_slug': 'beldex',
       'rank': 84,
       'circulating_supply': 980222595.0,
       'total_supply': 1400222610.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0725676935,
         'volume_24h': 645874.453410222,
         'market_cap': 71132493.0,
         'percent_change_1h': -0.07,
         'percent_change_24h': -0.63,
         'percent_change_7d': 1.59}},
       'last_updated': 1567986970},
      '3404': {'id': 3404,
       'name': 'Wixlar',
       'symbol': 'WIX',
       'website_slug': 'wixlar',
       'rank': 85,
       'circulating_supply': 2391612688.0,
       'total_supply': 5330000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0297168562,
         'volume_24h': 11947.6956895395,
         'market_cap': 71071210.0,
         'percent_change_1h': -1.28,
         'percent_change_24h': -0.27,
         'percent_change_7d': -8.19}},
       'last_updated': 1567986968},
      '693': {'id': 693,
       'name': 'Verge',
       'symbol': 'XVG',
       'website_slug': 'verge',
       'rank': 86,
       'circulating_supply': 15926617239.0,
       'total_supply': 15926617239.0,
       'max_supply': 16555000000.0,
       'quotes': {'USD': {'price': 0.0044509986,
         'volume_24h': 888876.63195651,
         'market_cap': 70889352.0,
         'percent_change_1h': -1.31,
         'percent_change_24h': -0.57,
         'percent_change_7d': -1.97}},
       'last_updated': 1567986961},
      '3224': {'id': 3224,
       'name': 'Qubitica',
       'symbol': 'QBIT',
       'website_slug': 'qubitica',
       'rank': 87,
       'circulating_supply': 2085316.0,
       'total_supply': 10000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 33.2411930237,
         'volume_24h': 112345.45388573,
         'market_cap': 69318398.0,
         'percent_change_1h': -0.02,
         'percent_change_24h': 1.63,
         'percent_change_7d': 6.2}},
       'last_updated': 1567986969},
      '3701': {'id': 3701,
       'name': 'RIF Token',
       'symbol': 'RIF',
       'website_slug': 'rif-token',
       'rank': 88,
       'circulating_supply': 477980957.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1429427336,
         'volume_24h': 3871514.03183582,
         'market_cap': 68323905.0,
         'percent_change_1h': 0.02,
         'percent_change_24h': 0.12,
         'percent_change_7d': 9.4}},
       'last_updated': 1567986969},
      '3115': {'id': 3115,
       'name': 'Maximine Coin',
       'symbol': 'MXM',
       'website_slug': 'maximine-coin',
       'rank': 89,
       'circulating_supply': 1649000000.0,
       'total_supply': 16000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0412529629,
         'volume_24h': 6442024.19962655,
         'market_cap': 68026136.0,
         'percent_change_1h': -0.09,
         'percent_change_24h': 7.58,
         'percent_change_7d': 49.91}},
       'last_updated': 1567986966},
      '3155': {'id': 3155,
       'name': 'Quant',
       'symbol': 'QNT',
       'website_slug': 'quant',
       'rank': 90,
       'circulating_supply': 12072738.0,
       'total_supply': 14612493.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 5.5157049226,
         'volume_24h': 2631297.89156621,
         'market_cap': 66589660.0,
         'percent_change_1h': -0.76,
         'percent_change_24h': 4.64,
         'percent_change_7d': -1.46}},
       'last_updated': 1567986967},
      '2130': {'id': 2130,
       'name': 'Enjin Coin',
       'symbol': 'ENJ',
       'website_slug': 'enjin-coin',
       'rank': 91,
       'circulating_supply': 776341213.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0827166271,
         'volume_24h': 4337452.38460489,
         'market_cap': 64216327.0,
         'percent_change_1h': 0.35,
         'percent_change_24h': 2.26,
         'percent_change_7d': 11.96}},
       'last_updated': 1567986964},
      '2469': {'id': 2469,
       'name': 'Zilliqa',
       'symbol': 'ZIL',
       'website_slug': 'zilliqa',
       'rank': 92,
       'circulating_supply': 8687360058.0,
       'total_supply': 12533042435.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.00694038,
         'volume_24h': 6992954.53528866,
         'market_cap': 60293580.0,
         'percent_change_1h': 0.76,
         'percent_change_24h': 0.67,
         'percent_change_7d': -1.11}},
       'last_updated': 1567986965},
      '1700': {'id': 1700,
       'name': 'Aeternity',
       'symbol': 'AE',
       'website_slug': 'aeternity',
       'rank': 93,
       'circulating_supply': 282265576.0,
       'total_supply': 328086519.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.2090029444,
         'volume_24h': 41958326.4424208,
         'market_cap': 58994336.0,
         'percent_change_1h': -1.72,
         'percent_change_24h': 2.02,
         'percent_change_7d': 1.5}},
       'last_updated': 1567986963},
      '3620': {'id': 3620,
       'name': 'Atlas Protocol',
       'symbol': 'ATP',
       'website_slug': 'atlas-protocol',
       'rank': 94,
       'circulating_supply': 2576065703.0,
       'total_supply': 4000000000.0,
       'max_supply': 10000000000.0,
       'quotes': {'USD': {'price': 0.022328362,
         'volume_24h': 2744693.46776034,
         'market_cap': 57519328.0,
         'percent_change_1h': -7.17,
         'percent_change_24h': -18.41,
         'percent_change_7d': -34.38}},
       'last_updated': 1567986968},
      '1230': {'id': 1230,
       'name': 'Steem',
       'symbol': 'STEEM',
       'website_slug': 'steem',
       'rank': 95,
       'circulating_supply': 343908578.0,
       'total_supply': 360882672.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.1672453177,
         'volume_24h': 454962.127464183,
         'market_cap': 57517099.0,
         'percent_change_1h': 0.03,
         'percent_change_24h': -1.0,
         'percent_change_7d': 1.13}},
       'last_updated': 1567986962},
      '1320': {'id': 1320,
       'name': 'Ardor',
       'symbol': 'ARDR',
       'website_slug': 'ardor',
       'rank': 96,
       'circulating_supply': 998999495.0,
       'total_supply': 998999495.0,
       'max_supply': 998999495.0,
       'quotes': {'USD': {'price': 0.0564249888,
         'volume_24h': 542616.680773218,
         'market_cap': 56368535.0,
         'percent_change_1h': -0.84,
         'percent_change_24h': 0.4,
         'percent_change_7d': 3.01}},
       'last_updated': 1567986962},
      '1455': {'id': 1455,
       'name': 'Golem',
       'symbol': 'GNT',
       'website_slug': 'golem-network-tokens',
       'rank': 97,
       'circulating_supply': 964450000.0,
       'total_supply': 1000000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0578834125,
         'volume_24h': 1661097.5758074,
         'market_cap': 55825657.0,
         'percent_change_1h': 0.37,
         'percent_change_24h': -1.38,
         'percent_change_7d': -0.63}},
       'last_updated': 1567986962},
      '1759': {'id': 1759,
       'name': 'Status',
       'symbol': 'SNT',
       'website_slug': 'status',
       'rank': 98,
       'circulating_supply': 3470483788.0,
       'total_supply': 6804870174.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 0.0157563616,
         'volume_24h': 13746885.0821321,
         'market_cap': 54682198.0,
         'percent_change_1h': 0.25,
         'percent_change_24h': 1.4,
         'percent_change_7d': 1.92}},
       'last_updated': 1567986963},
      '3897': {'id': 3897,
       'name': 'OKB',
       'symbol': 'OKB',
       'website_slug': 'okb',
       'rank': 99,
       'circulating_supply': 20000000.0,
       'total_supply': 300000000.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 2.6763215667,
         'volume_24h': 66563903.3675896,
         'market_cap': 53526431.0,
         'percent_change_1h': 0.08,
         'percent_change_24h': -0.95,
         'percent_change_7d': 7.68}},
       'last_updated': 1567986970},
      '1776': {'id': 1776,
       'name': 'Crypto.com',
       'symbol': 'MCO',
       'website_slug': 'crypto-com',
       'rank': 100,
       'circulating_supply': 15793831.0,
       'total_supply': 31587682.0,
       'max_supply': None,
       'quotes': {'USD': {'price': 3.2655550037,
         'volume_24h': 4123773.4410909,
         'market_cap': 51575624.0,
         'percent_change_1h': 0.38,
         'percent_change_24h': 1.48,
         'percent_change_7d': 0.35}},
       'last_updated': 1567986962}},
     'metadata': {'timestamp': 1567986611,
      'warning': 'WARNING: This API is now deprecated and will be taken offline soon.  Please switch to the new CoinMarketCap API to avoid interruptions in service. (https://pro.coinmarketcap.com/migrate/)',
      'num_cryptocurrencies': 2353,
      'error': None},
     'cached': True}




```python
data = pd.DataFrame(cap.ticker(start=0, limit=100)['data']).T
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>circulating_supply</th>
      <th>id</th>
      <th>last_updated</th>
      <th>max_supply</th>
      <th>name</th>
      <th>quotes</th>
      <th>rank</th>
      <th>symbol</th>
      <th>total_supply</th>
      <th>website_slug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.79241e+07</td>
      <td>1</td>
      <td>1567987170</td>
      <td>2.1e+07</td>
      <td>Bitcoin</td>
      <td>{'USD': {'price': 10443.2285007, 'volume_24h':...</td>
      <td>1</td>
      <td>BTC</td>
      <td>1.79241e+07</td>
      <td>bitcoin</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>1.07663e+08</td>
      <td>1027</td>
      <td>1567987163</td>
      <td>None</td>
      <td>Ethereum</td>
      <td>{'USD': {'price': 181.362484764, 'volume_24h':...</td>
      <td>2</td>
      <td>ETH</td>
      <td>1.07663e+08</td>
      <td>ethereum</td>
    </tr>
    <tr>
      <th>52</th>
      <td>4.29847e+10</td>
      <td>52</td>
      <td>1567987145</td>
      <td>1e+11</td>
      <td>XRP</td>
      <td>{'USD': {'price': 0.2632474231, 'volume_24h': ...</td>
      <td>3</td>
      <td>XRP</td>
      <td>9.99914e+10</td>
      <td>ripple</td>
    </tr>
    <tr>
      <th>1831</th>
      <td>1.79926e+07</td>
      <td>1831</td>
      <td>1567987147</td>
      <td>2.1e+07</td>
      <td>Bitcoin Cash</td>
      <td>{'USD': {'price': 307.807825121, 'volume_24h':...</td>
      <td>4</td>
      <td>BCH</td>
      <td>1.79926e+07</td>
      <td>bitcoin-cash</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.32126e+07</td>
      <td>2</td>
      <td>1567987145</td>
      <td>8.4e+07</td>
      <td>Litecoin</td>
      <td>{'USD': {'price': 70.619391981, 'volume_24h': ...</td>
      <td>5</td>
      <td>LTC</td>
      <td>6.32126e+07</td>
      <td>litecoin</td>
    </tr>
    <tr>
      <th>825</th>
      <td>4.07119e+09</td>
      <td>825</td>
      <td>1567987157</td>
      <td>None</td>
      <td>Tether</td>
      <td>{'USD': {'price': 1.003200803, 'volume_24h': 1...</td>
      <td>6</td>
      <td>USDT</td>
      <td>4.16006e+09</td>
      <td>tether</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>9.30917e+08</td>
      <td>1765</td>
      <td>1567987146</td>
      <td>None</td>
      <td>EOS</td>
      <td>{'USD': {'price': 3.785358539, 'volume_24h': 2...</td>
      <td>7</td>
      <td>EOS</td>
      <td>1.02762e+09</td>
      <td>eos</td>
    </tr>
    <tr>
      <th>1839</th>
      <td>1.55537e+08</td>
      <td>1839</td>
      <td>1567987145</td>
      <td>1.87537e+08</td>
      <td>Binance Coin</td>
      <td>{'USD': {'price': 22.5690225633, 'volume_24h':...</td>
      <td>8</td>
      <td>BNB</td>
      <td>1.87537e+08</td>
      <td>binance-coin</td>
    </tr>
    <tr>
      <th>3602</th>
      <td>1.7855e+07</td>
      <td>3602</td>
      <td>1567987150</td>
      <td>2.1e+07</td>
      <td>Bitcoin SV</td>
      <td>{'USD': {'price': 135.532445359, 'volume_24h':...</td>
      <td>9</td>
      <td>BSV</td>
      <td>1.7855e+07</td>
      <td>bitcoin-sv</td>
    </tr>
    <tr>
      <th>328</th>
      <td>1.71984e+07</td>
      <td>328</td>
      <td>1567987142</td>
      <td>None</td>
      <td>Monero</td>
      <td>{'USD': {'price': 77.4046423109, 'volume_24h':...</td>
      <td>10</td>
      <td>XMR</td>
      <td>1.71984e+07</td>
      <td>monero</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>2.59271e+10</td>
      <td>2010</td>
      <td>1567987144</td>
      <td>4.5e+10</td>
      <td>Cardano</td>
      <td>{'USD': {'price': 0.0467058741, 'volume_24h': ...</td>
      <td>11</td>
      <td>ADA</td>
      <td>3.11125e+10</td>
      <td>cardano</td>
    </tr>
    <tr>
      <th>512</th>
      <td>1.97474e+10</td>
      <td>512</td>
      <td>1567987143</td>
      <td>None</td>
      <td>Stellar</td>
      <td>{'USD': {'price': 0.0608908849, 'volume_24h': ...</td>
      <td>12</td>
      <td>XLM</td>
      <td>1.05303e+11</td>
      <td>stellar</td>
    </tr>
    <tr>
      <th>3957</th>
      <td>9.99499e+08</td>
      <td>3957</td>
      <td>1567987151</td>
      <td>None</td>
      <td>UNUS SED LEO</td>
      <td>{'USD': {'price': 1.07029423, 'volume_24h': 51...</td>
      <td>13</td>
      <td>LEO</td>
      <td>9.99499e+08</td>
      <td>unus-sed-leo</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>6.66821e+10</td>
      <td>1958</td>
      <td>1567987146</td>
      <td>None</td>
      <td>TRON</td>
      <td>{'USD': {'price': 0.0158395522, 'volume_24h': ...</td>
      <td>14</td>
      <td>TRX</td>
      <td>9.92813e+10</td>
      <td>tron</td>
    </tr>
    <tr>
      <th>2502</th>
      <td>2.45881e+08</td>
      <td>2502</td>
      <td>1567987146</td>
      <td>None</td>
      <td>Huobi Token</td>
      <td>{'USD': {'price': 4.0544796472, 'volume_24h': ...</td>
      <td>15</td>
      <td>HT</td>
      <td>5e+08</td>
      <td>huobi-token</td>
    </tr>
    <tr>
      <th>131</th>
      <td>9.02863e+06</td>
      <td>131</td>
      <td>1567987143</td>
      <td>1.89e+07</td>
      <td>Dash</td>
      <td>{'USD': {'price': 87.7976383124, 'volume_24h':...</td>
      <td>16</td>
      <td>DASH</td>
      <td>9.02863e+06</td>
      <td>dash</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>1.13322e+08</td>
      <td>1321</td>
      <td>1567987144</td>
      <td>2.1e+08</td>
      <td>Ethereum Classic</td>
      <td>{'USD': {'price': 6.6749199553, 'volume_24h': ...</td>
      <td>17</td>
      <td>ETC</td>
      <td>1.13322e+08</td>
      <td>ethereum-classic</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>6.60374e+08</td>
      <td>2011</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Tezos</td>
      <td>{'USD': {'price': 1.0646236965, 'volume_24h': ...</td>
      <td>18</td>
      <td>XTZ</td>
      <td>8.01313e+08</td>
      <td>tezos</td>
    </tr>
    <tr>
      <th>1720</th>
      <td>2.77953e+09</td>
      <td>1720</td>
      <td>1567987143</td>
      <td>2.77953e+09</td>
      <td>IOTA</td>
      <td>{'USD': {'price': 0.2451704818, 'volume_24h': ...</td>
      <td>19</td>
      <td>MIOTA</td>
      <td>2.77953e+09</td>
      <td>iota</td>
    </tr>
    <tr>
      <th>1376</th>
      <td>7.05388e+07</td>
      <td>1376</td>
      <td>1567987144</td>
      <td>1e+08</td>
      <td>NEO</td>
      <td>{'USD': {'price': 9.3110464199, 'volume_24h': ...</td>
      <td>20</td>
      <td>NEO</td>
      <td>1e+08</td>
      <td>neo</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>3.5e+08</td>
      <td>1975</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Chainlink</td>
      <td>{'USD': {'price': 1.8331335749, 'volume_24h': ...</td>
      <td>21</td>
      <td>LINK</td>
      <td>1e+09</td>
      <td>chainlink</td>
    </tr>
    <tr>
      <th>3794</th>
      <td>1.90688e+08</td>
      <td>3794</td>
      <td>1567987151</td>
      <td>None</td>
      <td>Cosmos</td>
      <td>{'USD': {'price': 2.734102502, 'volume_24h': 1...</td>
      <td>22</td>
      <td>ATOM</td>
      <td>2.37928e+08</td>
      <td>cosmos</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>1e+06</td>
      <td>1518</td>
      <td>1567987143</td>
      <td>None</td>
      <td>Maker</td>
      <td>{'USD': {'price': 447.423523577, 'volume_24h':...</td>
      <td>23</td>
      <td>MKR</td>
      <td>1e+06</td>
      <td>maker</td>
    </tr>
    <tr>
      <th>3408</th>
      <td>4.42377e+08</td>
      <td>3408</td>
      <td>1567987150</td>
      <td>None</td>
      <td>USD Coin</td>
      <td>{'USD': {'price': 1.0031950728, 'volume_24h': ...</td>
      <td>24</td>
      <td>USDC</td>
      <td>4.43564e+08</td>
      <td>usd-coin</td>
    </tr>
    <tr>
      <th>873</th>
      <td>9e+09</td>
      <td>873</td>
      <td>1567987143</td>
      <td>None</td>
      <td>NEM</td>
      <td>{'USD': {'price': 0.0472902441, 'volume_24h': ...</td>
      <td>25</td>
      <td>XEM</td>
      <td>9e+09</td>
      <td>nem</td>
    </tr>
    <tr>
      <th>2566</th>
      <td>5.33384e+08</td>
      <td>2566</td>
      <td>1567987147</td>
      <td>None</td>
      <td>Ontology</td>
      <td>{'USD': {'price': 0.7587283864, 'volume_24h': ...</td>
      <td>26</td>
      <td>ONT</td>
      <td>1e+09</td>
      <td>ontology</td>
    </tr>
    <tr>
      <th>3635</th>
      <td>9.66438e+09</td>
      <td>3635</td>
      <td>1567987150</td>
      <td>None</td>
      <td>Crypto.com Chain</td>
      <td>{'USD': {'price': 0.0394636644, 'volume_24h': ...</td>
      <td>27</td>
      <td>CRO</td>
      <td>1e+11</td>
      <td>crypto-com-chain</td>
    </tr>
    <tr>
      <th>3085</th>
      <td>1.80003e+08</td>
      <td>3085</td>
      <td>1567987148</td>
      <td>None</td>
      <td>INO COIN</td>
      <td>{'USD': {'price': 2.0415062137, 'volume_24h': ...</td>
      <td>28</td>
      <td>INO</td>
      <td>1e+09</td>
      <td>ino-coin</td>
    </tr>
    <tr>
      <th>1437</th>
      <td>7.37193e+06</td>
      <td>1437</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Zcash</td>
      <td>{'USD': {'price': 48.3719172853, 'volume_24h':...</td>
      <td>29</td>
      <td>ZEC</td>
      <td>7.37193e+06</td>
      <td>zcash</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1.2112e+11</td>
      <td>74</td>
      <td>1567987146</td>
      <td>None</td>
      <td>Dogecoin</td>
      <td>{'USD': {'price': 0.0025088289, 'volume_24h': ...</td>
      <td>30</td>
      <td>DOGE</td>
      <td>1.2112e+11</td>
      <td>dogecoin</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2603</th>
      <td>2.35621e+11</td>
      <td>2603</td>
      <td>1567987146</td>
      <td>None</td>
      <td>Pundi X</td>
      <td>{'USD': {'price': 0.0003716197, 'volume_24h': ...</td>
      <td>71</td>
      <td>NPXS</td>
      <td>2.61835e+11</td>
      <td>pundi-x</td>
    </tr>
    <tr>
      <th>1521</th>
      <td>1.15832e+08</td>
      <td>1521</td>
      <td>1567987143</td>
      <td>2e+08</td>
      <td>Komodo</td>
      <td>{'USD': {'price': 0.73817598, 'volume_24h': 25...</td>
      <td>72</td>
      <td>KMD</td>
      <td>1.15832e+08</td>
      <td>komodo</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>9.347e+07</td>
      <td>3134</td>
      <td>1567987148</td>
      <td>None</td>
      <td>ETERNAL TOKEN</td>
      <td>{'USD': {'price': 0.9017733334, 'volume_24h': ...</td>
      <td>73</td>
      <td>XET</td>
      <td>2e+08</td>
      <td>eternal-token</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>1.2014e+10</td>
      <td>2405</td>
      <td>1567987146</td>
      <td>None</td>
      <td>IOST</td>
      <td>{'USD': {'price': 0.0069610728, 'volume_24h': ...</td>
      <td>74</td>
      <td>IOST</td>
      <td>2.1e+10</td>
      <td>iostoken</td>
    </tr>
    <tr>
      <th>3829</th>
      <td>3.61967e+07</td>
      <td>3829</td>
      <td>1567987151</td>
      <td>None</td>
      <td>Nash Exchange</td>
      <td>{'USD': {'price': 2.2427890172, 'volume_24h': ...</td>
      <td>75</td>
      <td>NEX</td>
      <td>5.62961e+07</td>
      <td>nash-exchange</td>
    </tr>
    <tr>
      <th>1866</th>
      <td>1.0025e+09</td>
      <td>1866</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Bytom</td>
      <td>{'USD': {'price': 0.08043945, 'volume_24h': 64...</td>
      <td>76</td>
      <td>BTM</td>
      <td>1.407e+09</td>
      <td>bytom</td>
    </tr>
    <tr>
      <th>3218</th>
      <td>2.03408e+07</td>
      <td>3218</td>
      <td>1567987149</td>
      <td>None</td>
      <td>Energi</td>
      <td>{'USD': {'price': 3.8660368807, 'volume_24h': ...</td>
      <td>77</td>
      <td>NRG</td>
      <td>2.03408e+07</td>
      <td>energi</td>
    </tr>
    <tr>
      <th>1042</th>
      <td>4.1817e+10</td>
      <td>1042</td>
      <td>1567987142</td>
      <td>None</td>
      <td>Siacoin</td>
      <td>{'USD': {'price': 0.001875081, 'volume_24h': 1...</td>
      <td>78</td>
      <td>SC</td>
      <td>4.1817e+10</td>
      <td>siacoin</td>
    </tr>
    <tr>
      <th>2308</th>
      <td>7.79051e+07</td>
      <td>2308</td>
      <td>1567987145</td>
      <td>None</td>
      <td>Dai</td>
      <td>{'USD': {'price': 1.0048955868, 'volume_24h': ...</td>
      <td>79</td>
      <td>DAI</td>
      <td>7.79051e+07</td>
      <td>dai</td>
    </tr>
    <tr>
      <th>3840</th>
      <td>2.2085e+07</td>
      <td>3840</td>
      <td>1567987151</td>
      <td>None</td>
      <td>1irstcoin</td>
      <td>{'USD': {'price': 3.5387006419, 'volume_24h': ...</td>
      <td>80</td>
      <td>FST</td>
      <td>1e+08</td>
      <td>1irstcoin</td>
    </tr>
    <tr>
      <th>4049</th>
      <td>3.68387e+08</td>
      <td>4049</td>
      <td>1567987152</td>
      <td>None</td>
      <td>Bitbook Gambling</td>
      <td>{'USD': {'price': 0.2107729991, 'volume_24h': ...</td>
      <td>81</td>
      <td>BXK</td>
      <td>7.41456e+08</td>
      <td>bitbook-gambling</td>
    </tr>
    <tr>
      <th>213</th>
      <td>6.57297e+07</td>
      <td>213</td>
      <td>1567987142</td>
      <td>None</td>
      <td>MonaCoin</td>
      <td>{'USD': {'price': 1.1246863004, 'volume_24h': ...</td>
      <td>82</td>
      <td>MONA</td>
      <td>6.57297e+07</td>
      <td>monacoin</td>
    </tr>
    <tr>
      <th>693</th>
      <td>1.59266e+10</td>
      <td>693</td>
      <td>1567987142</td>
      <td>1.6555e+10</td>
      <td>Verge</td>
      <td>{'USD': {'price': 0.0044719722, 'volume_24h': ...</td>
      <td>83</td>
      <td>XVG</td>
      <td>1.59266e+10</td>
      <td>verge</td>
    </tr>
    <tr>
      <th>3987</th>
      <td>9.80223e+08</td>
      <td>3987</td>
      <td>1567987151</td>
      <td>None</td>
      <td>Beldex</td>
      <td>{'USD': {'price': 0.0725997451, 'volume_24h': ...</td>
      <td>84</td>
      <td>BDX</td>
      <td>1.40022e+09</td>
      <td>beldex</td>
    </tr>
    <tr>
      <th>3812</th>
      <td>2.05864e+10</td>
      <td>3812</td>
      <td>1567987151</td>
      <td>1e+11</td>
      <td>Flexacoin</td>
      <td>{'USD': {'price': 0.0034562949, 'volume_24h': ...</td>
      <td>85</td>
      <td>FXC</td>
      <td>2.05864e+10</td>
      <td>flexacoin</td>
    </tr>
    <tr>
      <th>3404</th>
      <td>2.39161e+09</td>
      <td>3404</td>
      <td>1567987149</td>
      <td>None</td>
      <td>Wixlar</td>
      <td>{'USD': {'price': 0.0297168649, 'volume_24h': ...</td>
      <td>86</td>
      <td>WIX</td>
      <td>5.33e+09</td>
      <td>wixlar</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>2.08532e+06</td>
      <td>3224</td>
      <td>1567987149</td>
      <td>None</td>
      <td>Qubitica</td>
      <td>{'USD': {'price': 33.2392305568, 'volume_24h':...</td>
      <td>87</td>
      <td>QBIT</td>
      <td>1e+07</td>
      <td>qubitica</td>
    </tr>
    <tr>
      <th>3115</th>
      <td>1.649e+09</td>
      <td>3115</td>
      <td>1567987148</td>
      <td>None</td>
      <td>Maximine Coin</td>
      <td>{'USD': {'price': 0.0413027525, 'volume_24h': ...</td>
      <td>88</td>
      <td>MXM</td>
      <td>1.6e+10</td>
      <td>maximine-coin</td>
    </tr>
    <tr>
      <th>3701</th>
      <td>4.77981e+08</td>
      <td>3701</td>
      <td>1567987150</td>
      <td>None</td>
      <td>RIF Token</td>
      <td>{'USD': {'price': 0.1423420356, 'volume_24h': ...</td>
      <td>89</td>
      <td>RIF</td>
      <td>1e+09</td>
      <td>rif-token</td>
    </tr>
    <tr>
      <th>3155</th>
      <td>1.20727e+07</td>
      <td>3155</td>
      <td>1567987149</td>
      <td>None</td>
      <td>Quant</td>
      <td>{'USD': {'price': 5.52759386, 'volume_24h': 26...</td>
      <td>90</td>
      <td>QNT</td>
      <td>1.46125e+07</td>
      <td>quant</td>
    </tr>
    <tr>
      <th>2130</th>
      <td>7.76341e+08</td>
      <td>2130</td>
      <td>1567987146</td>
      <td>None</td>
      <td>Enjin Coin</td>
      <td>{'USD': {'price': 0.0827353878, 'volume_24h': ...</td>
      <td>91</td>
      <td>ENJ</td>
      <td>1e+09</td>
      <td>enjin-coin</td>
    </tr>
    <tr>
      <th>2469</th>
      <td>8.68736e+09</td>
      <td>2469</td>
      <td>1567987146</td>
      <td>None</td>
      <td>Zilliqa</td>
      <td>{'USD': {'price': 0.0069222843, 'volume_24h': ...</td>
      <td>92</td>
      <td>ZIL</td>
      <td>1.2533e+10</td>
      <td>zilliqa</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>2.82267e+08</td>
      <td>1700</td>
      <td>1567987143</td>
      <td>None</td>
      <td>Aeternity</td>
      <td>{'USD': {'price': 0.2091437806, 'volume_24h': ...</td>
      <td>93</td>
      <td>AE</td>
      <td>3.28088e+08</td>
      <td>aeternity</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>3.4391e+08</td>
      <td>1230</td>
      <td>1567987142</td>
      <td>None</td>
      <td>Steem</td>
      <td>{'USD': {'price': 0.1669576783, 'volume_24h': ...</td>
      <td>94</td>
      <td>STEEM</td>
      <td>3.60884e+08</td>
      <td>steem</td>
    </tr>
    <tr>
      <th>3620</th>
      <td>2.57607e+09</td>
      <td>3620</td>
      <td>1567987150</td>
      <td>1e+10</td>
      <td>Atlas Protocol</td>
      <td>{'USD': {'price': 0.0222864884, 'volume_24h': ...</td>
      <td>95</td>
      <td>ATP</td>
      <td>4e+09</td>
      <td>atlas-protocol</td>
    </tr>
    <tr>
      <th>1320</th>
      <td>9.98999e+08</td>
      <td>1320</td>
      <td>1567987142</td>
      <td>9.98999e+08</td>
      <td>Ardor</td>
      <td>{'USD': {'price': 0.0564993021, 'volume_24h': ...</td>
      <td>96</td>
      <td>ARDR</td>
      <td>9.98999e+08</td>
      <td>ardor</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>9.6445e+08</td>
      <td>1455</td>
      <td>1567987143</td>
      <td>None</td>
      <td>Golem</td>
      <td>{'USD': {'price': 0.0578334465, 'volume_24h': ...</td>
      <td>97</td>
      <td>GNT</td>
      <td>1e+09</td>
      <td>golem-network-tokens</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>3.47048e+09</td>
      <td>1759</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Status</td>
      <td>{'USD': {'price': 0.0157600024, 'volume_24h': ...</td>
      <td>98</td>
      <td>SNT</td>
      <td>6.80487e+09</td>
      <td>status</td>
    </tr>
    <tr>
      <th>3897</th>
      <td>2e+07</td>
      <td>3897</td>
      <td>1567987151</td>
      <td>None</td>
      <td>OKB</td>
      <td>{'USD': {'price': 2.6744782953, 'volume_24h': ...</td>
      <td>99</td>
      <td>OKB</td>
      <td>3e+08</td>
      <td>okb</td>
    </tr>
    <tr>
      <th>1776</th>
      <td>1.57938e+07</td>
      <td>1776</td>
      <td>1567987144</td>
      <td>None</td>
      <td>Crypto.com</td>
      <td>{'USD': {'price': 3.2511073239, 'volume_24h': ...</td>
      <td>100</td>
      <td>MCO</td>
      <td>3.15877e+07</td>
      <td>crypto-com</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 10 columns</p>
</div>




```python
data['quotes'][0]
```




    {'USD': {'price': 10443.2285007,
      'volume_24h': 13673859386.5884,
      'market_cap': 187185075127.0,
      'percent_change_1h': 0.02,
      'percent_change_24h': -0.79,
      'percent_change_7d': 7.38}}




```python
data['quotes'][0]['USD']

```




    {'price': 10443.2285007,
     'volume_24h': 13673859386.5884,
     'market_cap': 187185075127.0,
     'percent_change_1h': 0.02,
     'percent_change_24h': -0.79,
     'percent_change_7d': 7.38}




```python
data['quotes'][0]['USD']['market_cap']
```




    187185075127.0




```python
data['Market_Cap'] = data['quotes'].apply(lambda x: x['USD']['market_cap'])
data['Price'] = data['quotes'].apply(lambda x: x['USD']['price'])

```


```python

```


```python

```

### JSON：
JSON (JavaScript Object Notation) is a lightweight data-interchange format. It is easy for humans to read and write. It is easy for machines to parse and generate.

**json** library is needed so that we can work with the JSON content we get from the API. In this case, we get a dictionary for each Channel’s information such as name, id, views and other information.

### <font color="green">Q: Where do we see JSON objects the most?</font>


```python

```


```python
import json
```


```python
zoo = """
{
    "zoo_animal":"Lion",
    "food":["Meat", "Veggies", "Honey"],
    "fur":"Golden",
    "clothes": null,
    "diet":[{"zoo_animal":" Gazelle", "food":"grass", "fur":"Brown"}]
}
"""
```


```python
import pandas as pd 
```


```python
json.loads(zoo)
```




    {'zoo_animal': 'Lion',
     'food': ['Meat', 'Veggies', 'Honey'],
     'fur': 'Golden',
     'clothes': None,
     'diet': [{'zoo_animal': ' Gazelle', 'food': 'grass', 'fur': 'Brown'}]}




```python
data = json.loads(zoo)
```


```python
data
```




    {'zoo_animal': 'Lion',
     'food': ['Meat', 'Veggies', 'Honey'],
     'fur': 'Golden',
     'clothes': None,
     'diet': [{'zoo_animal': ' Gazelle', 'food': 'grass', 'fur': 'Brown'}]}




```python
json.dumps(data)
```




    '{"zoo_animal": "Lion", "food": ["Meat", "Veggies", "Honey"], "fur": "Golden", "clothes": null, "diet": [{"zoo_animal": " Gazelle", "food": "grass", "fur": "Brown"}]}'




```python

dframe = pd.DataFrame(data['diet'])
```


```python
dframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food</th>
      <th>fur</th>
      <th>zoo_animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>grass</td>
      <td>Brown</td>
      <td>Gazelle</td>
    </tr>
  </tbody>
</table>
</div>



## Data Munging


```python
import pandas as pd
```


```python
players = pd.read_csv('combine.csv')
```


```python
players.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>name</th>
      <th>firstname</th>
      <th>lastname</th>
      <th>position</th>
      <th>heightfeet</th>
      <th>heightinches</th>
      <th>heightinchestotal</th>
      <th>weight</th>
      <th>arms</th>
      <th>...</th>
      <th>vertical</th>
      <th>broad</th>
      <th>bench</th>
      <th>round</th>
      <th>college</th>
      <th>pick</th>
      <th>pickround</th>
      <th>picktotal</th>
      <th>wonderlic</th>
      <th>nflgrade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>Ameer Abdullah</td>
      <td>Ameer</td>
      <td>Abdullah</td>
      <td>RB</td>
      <td>5</td>
      <td>9.0</td>
      <td>69.0</td>
      <td>205</td>
      <td>0.0</td>
      <td>...</td>
      <td>42.5</td>
      <td>130</td>
      <td>24</td>
      <td>0</td>
      <td>Nebraska</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>Nelson Agholor</td>
      <td>Nelson</td>
      <td>Agholor</td>
      <td>WR</td>
      <td>6</td>
      <td>0.0</td>
      <td>72.0</td>
      <td>198</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>USC</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>Jay Ajayi</td>
      <td>Jay</td>
      <td>Ajayi</td>
      <td>RB</td>
      <td>6</td>
      <td>0.0</td>
      <td>72.0</td>
      <td>221</td>
      <td>0.0</td>
      <td>...</td>
      <td>39.0</td>
      <td>121</td>
      <td>19</td>
      <td>0</td>
      <td>Boise St.</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>Kwon Alexander</td>
      <td>Kwon</td>
      <td>Alexander</td>
      <td>OLB</td>
      <td>6</td>
      <td>1.0</td>
      <td>73.0</td>
      <td>227</td>
      <td>0.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>121</td>
      <td>24</td>
      <td>0</td>
      <td>LSU</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>Mario Alford</td>
      <td>Mario</td>
      <td>Alford</td>
      <td>WR</td>
      <td>5</td>
      <td>8.0</td>
      <td>68.0</td>
      <td>180</td>
      <td>0.0</td>
      <td>...</td>
      <td>34.0</td>
      <td>121</td>
      <td>13</td>
      <td>0</td>
      <td>West Virginia</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
players.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4947 entries, 0 to 4946
    Data columns (total 26 columns):
    year                 4947 non-null int64
    name                 4947 non-null object
    firstname            4947 non-null object
    lastname             4947 non-null object
    position             4947 non-null object
    heightfeet           4947 non-null int64
    heightinches         4947 non-null float64
    heightinchestotal    4947 non-null float64
    weight               4947 non-null int64
    arms                 4947 non-null float64
    hands                4947 non-null float64
    fortyyd              4947 non-null float64
    twentyyd             4947 non-null float64
    tenyd                4947 non-null float64
    twentyss             4947 non-null float64
    threecone            4947 non-null float64
    vertical             4947 non-null float64
    broad                4947 non-null int64
    bench                4947 non-null int64
    round                4947 non-null int64
    college              3477 non-null object
    pick                 3156 non-null object
    pickround            4947 non-null int64
    picktotal            4947 non-null int64
    wonderlic            4947 non-null int64
    nflgrade             4947 non-null float64
    dtypes: float64(11), int64(9), object(6)
    memory usage: 1004.9+ KB



```python
players.isnull().sum()
```




    year                    0
    name                    0
    firstname               0
    lastname                0
    position                0
    heightfeet              0
    heightinches            0
    heightinchestotal       0
    weight                  0
    arms                    0
    hands                   0
    fortyyd                 0
    twentyyd                0
    tenyd                   0
    twentyss                0
    threecone               0
    vertical                0
    broad                   0
    bench                   0
    round                   0
    college              1470
    pick                 1791
    pickround               0
    picktotal               0
    wonderlic               0
    nflgrade                0
    dtype: int64




```python
missing=players.isnull().sum()
missing[missing > 0]
```




    college    1470
    pick       1791
    dtype: int64




```python
players['college'].fillna('Pending', inplace=True)

```


```python
playersDropped = players.dropna(subset = ['pick'])
```


```python
playersDropped.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3156 entries, 322 to 4946
    Data columns (total 26 columns):
    year                 3156 non-null int64
    name                 3156 non-null object
    firstname            3156 non-null object
    lastname             3156 non-null object
    position             3156 non-null object
    heightfeet           3156 non-null int64
    heightinches         3156 non-null float64
    heightinchestotal    3156 non-null float64
    weight               3156 non-null int64
    arms                 3156 non-null float64
    hands                3156 non-null float64
    fortyyd              3156 non-null float64
    twentyyd             3156 non-null float64
    tenyd                3156 non-null float64
    twentyss             3156 non-null float64
    threecone            3156 non-null float64
    vertical             3156 non-null float64
    broad                3156 non-null int64
    bench                3156 non-null int64
    round                3156 non-null int64
    college              3155 non-null object
    pick                 3156 non-null object
    pickround            3156 non-null int64
    picktotal            3156 non-null int64
    wonderlic            3156 non-null int64
    nflgrade             3156 non-null float64
    dtypes: float64(11), int64(9), object(6)
    memory usage: 665.7+ KB



```python
len(players) - len(playersDropped)
```




    1791




```python
(len(players) - len(playersDropped)) / len(players)
```




    0.36203759854457246




```python
players.drop(['pick'], axis = 1, inplace=True)
```


```python
del players['pick']
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-8273623369cb> in <module>
    ----> 1 del (players['pick'])
    

    NameError: name 'players' is not defined



```python
players['position'].unique()
```




    array(['RB', 'WR', 'OLB', 'FS', 'DE', 'TE', 'ILB', 'DT', 'P', 'QB', 'OG',
           'OT', 'K', 'FB', 'SS', 'LS', 'CB', 'C', 'NT', 'OC'], dtype=object)




```python
players['position'].replace(['RB', 'QB'], ['Running Back', 'Quarter Back'], inplace=True)
```


```python
positions = pd.get_dummies(players['position'], prefix = 'Pos')
```


```python
positions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos_C</th>
      <th>Pos_CB</th>
      <th>Pos_DE</th>
      <th>Pos_DT</th>
      <th>Pos_FB</th>
      <th>Pos_FS</th>
      <th>Pos_ILB</th>
      <th>Pos_K</th>
      <th>Pos_LS</th>
      <th>Pos_NT</th>
      <th>Pos_OC</th>
      <th>Pos_OG</th>
      <th>Pos_OLB</th>
      <th>Pos_OT</th>
      <th>Pos_P</th>
      <th>Pos_Quarter Back</th>
      <th>Pos_Running Back</th>
      <th>Pos_SS</th>
      <th>Pos_TE</th>
      <th>Pos_WR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
data = pd.merge(players, positions, left_index = True, right_index = True)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-17ca4cc4c1e9> in <module>
    ----> 1 data = pd.merge(players, positions, left_index = True, right_index = True)
    

    NameError: name 'pd' is not defined



```python
data
```


```python
#d.set_option('display.max_columns', 100)
```


```python
positions = pd.get_dummies(players['position'], prefix = 'Pos')
```


```python
postions.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-97-1dda302753cf> in <module>
    ----> 1 postions.head()
    

    NameError: name 'postions' is not defined



```python

```


```python

```


```python

```

### <font color="green">Q: What is the equivalent of a JSON object in Python?</font>


```python

```

# <font color="red">IV. Hack Project: Election Day </font>

In this project, we will analyze Poll datasets.
1. Who was being polled and what was their party affiliation?
2. Did the poll results favor Romney or Obama?
3. How do undecided voters effect the poll?
4. Can we account for the undecided voters?
5. How did voter sentiment change over time?
6. Can we see an effect in the polls from the debates?


```python

```

## Extra Homework 
* p-value
* hypothesis testing
* student t distribution
* chi-squared distribution
* poisson distribution
* binomial dist
* gaussion dist
* Gamma dist
* Uniform dist


```python
df.info
df.describe
df.corr()
df.isnull().sum()
```
