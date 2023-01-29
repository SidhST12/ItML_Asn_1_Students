import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets
from scipy import stats

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
plt.rcParams["figure.figsize"] = (8,8)
#pip install missingno
class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure
    def missingvalues(self): #print missing values in dataframe
        if self.data.isnull().any(axis=None):
           print("\nPreview of data with null values:\nxxxxxxxxxxxxx")
           print(self.data[self.data.isnull().any(axis=1)].head(3))
           missingno.matrix(self.data)
        else:
           print("no Missing values found")
    def duplicatedvalues(self):#print duplicate values in data
        if len(self.data[self.data.duplicated()]) > 0:
            print("No. of duplicated entries: ", len(self.data[self.data.duplicated()]))
            print(self.data[self.data.duplicated(keep=False)].sort_values(by=list(self.data.columns)).head())
        else:
            print("No duplicated entries found")
    def top5(self):
    #Given dataframe, generate top 5 unique values for non-numeric data"""
        columns = self.data.select_dtypes(include=['object', 'category']).columns
        for col in columns:
            print("Top 5 unique values of " + col)
            print(self.data[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"})[
                 :min(5, len(self.data[col].value_counts()))])
            print(" ")
    def correlation(self): #show correlations between different numeric variables
        corr_matrix= self.data.corr()
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix,center=0, linewidths=.5, annot=True, cmap="YlGnBu", yticklabels=True,mask=mask)
        plt.show()
    def BasicStats(self):#prints basic statistics
        return self.data.describe()
    def Outliers(self):#print outliers 
        #setting threshold value
        threshold = 3
        columns= self.data.select_dtypes(include=['int','float']).columns
        for col in columns:
            z = np.abs(stats.zscore(self.data[col]))
        #extracting indices of the outliers
            outliers = np.where(z > threshold)
            print(outliers)
    def boxplot(self):#plot boxplot to detect outliers
        self.data.boxplot(grid=False, rot=45,fontsize=15)

        

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 =widgets.Output()
        out10= widgets.Output()

    

        tab = widgets.Tab(children = [out1, out2, out3,out4,out5,out6,out7,out8,out9,out10])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        tab.set_title(3,'Missing values')
        tab.set_title(4,'duplicated Values')
        tab.set_title(5,'top 5 values for categories')
        tab.set_title(6,'Correlations')
        tab.set_title(7,'Basic Stats')
        tab.set_title(8,'Outliers')
        tab.set_title(9,'Boxplot')
        display(tab)

    

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)
        with out4:
            fig4 =self.missingvalues()
            plt.show(fig4)
        with out5:
            self.duplicatedvalues()
        with out6:
            self.top5()
        with out7:
            fig5 =self.correlation()
            plt.show(fig5)
        with out8:
            print(self.BasicStats())
        with out9:
            self.Outliers()
        with out10:
            fig6= self.boxplot()
            plt.show(fig6)