import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

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
    def des(self):
        for col in self.num:
            print(col)
            print(self.data[col].describe())
            print('\n \n')   

    def nan_values(self): 
        for col in self.num:
            print(col)
            print('Numbers of values:',self.data[col].count())
            print('Numbers of the nan values:', self.data[col].isnull().sum())
            self.data[col]=self.data[col].dropna(axis=0)
            print('Numbers of values after droping nan:',self.data[col].count())
            print('\n \n') 

    def outliers(self):
        for col in self.num:
            q1,q3=self.data[col].quantile([0.25,0.76])
            IQR=q3-q1
            lower_range= q1-(1.5*IQR)
            upper_range= q3+(1.5*IQR)
            outliers = self.data[col][((self.data[col]<lower_range) | (self.data[col]>upper_range))]
            print(col)
            print('lower range of outliers: ', lower_range)
            print('upper range of outliers: ', upper_range)
            print('number of outliers: '+ str(len(outliers)))
            print('max outlier value: '+ str(outliers.max()))
            print('min outlier value: '+ str(outliers.min())+'\n')

    def pairplot(self,splitTarg=False, show=True):
        for col in self.num:
            if splitTarg == False:
                sns.pairplot(data=self.data, kind="reg")
            if splitTarg == True:
                sns.pairplot(data=self.data, hue=self.target,kind="reg")
            if show==True:
                plt.show()
            return plt

    def correlation(self, show=True):
        data=self.data.apply(pd.to_numeric, errors='coerce')
        data=data.drop(columns=self.cat)
        data=data.corr()
        mask=np.triu(np.ones_like(data, dtype=bool))
        sns.heatmap(data, center=0, linewidths=0.5, annot=True, cmap="YlGnBu", yticklabels=True, mask=mask)
        if show==True:
            plt.show()
        return plt

    def target_balance(self):
        return self.data[self.target].value_counts()

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8, out9])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Describe')
        tab.set_title(2, 'nan value')
        tab.set_title(3, 'Categorical')
        tab.set_title(4, 'Numerical')
        tab.set_title(5, 'Outliers')
        tab.set_title(6, 'Pairolot')
        tab.set_title(7, 'Correlation')
        tab.set_title(8, 'Target Balance')
        display(tab)

        with out1:
            self.info()

        with out2:
            self.des()

        with out3:
            self.nan_values()   

        with out4:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out5:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)
            
        with out6:
            self.outliers()

        with out7:
            fig4 = self.pairplot()
            plt.show(fig4)

        with out8:
            fig5 = self.correlation()
            plt.show(fig5)
        
        with out9:
            print("Target count:")
            display(self.target_balance())
        

