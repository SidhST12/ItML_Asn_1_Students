import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

"""""""""""""""
When you are ready to run my code at the very least know the proper steps so you can get it running! 
I tried to make it as easy as possible. 

All you need to do is set a target column, for example..
df_eda = ml_utils.edaDF(df,"Cholesterol")
print(df_eda.giveTarget())

Then just call the EDA function 
df_eda.fullEDA(k=1.5, scatterplot=False, optional_countplots=False, optional_histplots=False) 
No need for setNum or setCat!!

Please note the new parameters K, Scatterplot, optional_countplots and optional_countplots. 

K Value sets how many times the threshold is set to cut off outliers. So for example if you have it set to 1.5,
then the method will filter outliers 1.5x below the 25th percentile and 1.5x above the 75th percentile for every column. 

Set scatterplot to either True/False depending on whether you want to see scatterplots of the target variable 

Set optional_countplots to True if your target is categorical and False if it is numerical. And set optional_histplots to True if your
target is numerical and False if it is categorical. It will take a long time to run if you dont get this right as the countplots
and histplots method will sort each numerical and categorical value into its own bin.
"""""""""""""""

class edaDF:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.autoSetTypes()

    def info(self):
        """
    Method to display the dataframe information   
    """ 
        return self.data.info()

    def giveTarget(self):
        """
    Method to return the target variable
    """
        return self.target

    def autoSetTypes(self):
        """
    Automatically identify categorical and numerical columns in the dataframe
    """
        self.cat = self.data.select_dtypes(include=['object', 'category']).columns.tolist()# saves categorical columns as a list in 'cat'
        self.num = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()# saves numerical columns as a list in 'num'

        if self.cat.count(self.target) > 0:
            self.cat.remove(self.target)
        if self.num.count(self.target) > 0:
            self.num.remove(self.target)

    def describe(self):
        """
    Method to display the summary statistics of numerical columns
    """
        return self.data[self.num].describe()


    def valueCounts(self, columns=None):
        """
    Method to display the value counts for categorical columns
    """
        if columns is None:
            columns = self.cat
        return self.data[columns].apply(lambda x: x.value_counts())

    def countPlots(self, splitTarg=False, show=True, optional = True):
        """
        Method to create count plots for categorical columns
        """
        if optional == False or len(self.cat) == 0: # if optional is set to False, the method will not execute
            return
        n = len(self.cat) # number of categorical columns
        cols = 2 # number of columns in the plot
        figure, ax = plt.subplots (nrows=math.ceil(n/cols), ncols=cols, figsize=(20, 20), gridspec_kw={'wspace': 0.3}) # creating subplots
        r = 0
        c = 0
        for col in self.cat: # loop through each categorical column
            if splitTarg == False: # if splitTarg is set to False, the plot will be a simple countplot
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True: # if splitTarg is set to True, the plot will be a countplot with target variable as hue
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1 # increment column
            if c == cols: # if column is equal to the number of columns in the plot
                r += 1 # increment row
                c = 0 # reset column to 0
        if show == True:
            figure.tight_layout() # adjust the layout of the plot
            figure.show() # show the plot
        return figure # return the figure

    def histPlot(self, kde=True, splitTarg=False, show=True, optional = True):
        """
        Method to create histograms for numerical columns
        """
        if optional == False: # if optional is set to False, the method will not execute
            return
        n = len(self.num) # number of numerical columns
        cols = 2 # number of columns in the plot
        figure, ax = plt.subplots(math.ceil(n/cols), cols, figsize=(20, 20), gridspec_kw={'wspace': 0.3}) # creating subplots
        r = 0
        c = 0
        for col in self.num: # loop through each numerical column
            if splitTarg == False: # if splitTarg is set to False, the plot will be a simple histogram
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True: # if splitTarg is set to True, the plot will be a histogram with target variable as hue
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1 # increment column
            if c == cols: # if column is equal to the number of columns in the plot
                r += 1 # increment row
                c = 0 # reset column to 0
        if show == True:
            figure.tight_layout() # adjust the layout of the plot
            figure.show() # show the plot
        return figure # return the figure

    def displayOutliers(self, k):
        """
        Method to display the potential outliers in numerical columns
        """
        for col in self.num:
            q1 = self.data[col].quantile(0.25) # first quartile
            q3 = self.data[col].quantile(0.75) # third quartile
            iqr = q3 - q1 # interquartile range
            lower_bound = q1 - (k * iqr) # lower bound for outliers
            upper_bound = q3 + (k * iqr) # upper bound for outliers
            outliers = self.data.loc[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)] # find potential outliers
            if not outliers.empty: # if there are any potential outliers
                print(f"Potential outliers in column '{col}':") # print the column name
                display(outliers) # display the potential outliers

    def handleMissing(self,threshold = None, drop = False):
        """
    Handle missing values in the dataframe
    Parameters:
        threshold (float): threshold for the percentage of missing values
        method (str): method to handle missing values (mean, median, mode)
    """
        if threshold: # if threshold is set
            if drop: # drop rows with missing values above the threshold
                self.data = self.data.dropna(thresh = threshold) # calculate the percentage of missing values in each column
            else:
                self.data = self.data.fillna(self.data.mean()) # fill missing values with the mean
        else:
            self.data = self.data.fillna(self.data.mean()) # if threshold is not set, fill missing values with the mean

    def correlationMatrix(self, data=None): 
        """
        Method to create a correlation matrix
    """
        if data is None:
            data = self.data # if data is not provided, use the data in the class
        corr = data.corr() # calculate the correlation matrix
        sns.heatmap(corr, annot=True) # create a heatmap with annotations
        plt.show() # display the heatmap

    def targetDistribution(self, show=True):
        """
        Visualize the distribution of the target variable
        """
        figure, ax = plt.subplots() # create a figure and axis
        sns.histplot(data=self.data, x=self.target, kde=True, ax=ax) # create a histogram with kernel density estimation
        if show == True:
            figure.show() # show the histogram
        return figure # return the figure

    def featureVsTarget(self, scatterplot=True, show=True):
        """
        Visualize the relationship between the target variable and each feature
        """
        if scatterplot:
            n = len(self.num) #number of numerical columns
            cols = 2 # number of columns in the plot
            figure, ax = plt.subplots(math.ceil(n/cols), cols, figsize=(20, 20), gridspec_kw={'wspace': 0.3}) # creating subplots
            r = 0
            c = 0
            for col in self.num: # loop through each numerical column
                sns.scatterplot(data=self.data, x=col, y=self.target, ax=ax[r][c]) # create a scatterplot for the relationship between the target variable and the feature
                c += 1 # increment column
                if c == cols: # if column is equal to the number of columns in the plot
                    r += 1 # increment row
                    c = 0 # reset column to 0
            if show == True:
                figure.tight_layout() # adjust the layout of the plot
                figure.show() # show the plot
            return figure # return the figure

    def fullEDA(self, k, scatterplot=True, optional_countplots=True, optional_histplots=True):
        """
        # Method to run a full EDA on the dataframe
    """ 
        self.autoSetTypes() # automatically identify categorical and numerical columns

        out1 = widgets.Output() # create output widgets
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()
        out10 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]) # create a tab layout
        tab.set_title(0, 'Info') # set tab titles
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        tab.set_title(3, 'Handle Missing')
        tab.set_title(4, 'Summary stats')
        tab.set_title(5, 'Remove outliers')
        tab.set_title(6, 'Correlation matrix')
        tab.set_title(7, 'Target Distribution')
        tab.set_title(8, 'Feature VS. Target') 
        tab.set_title(9, 'Value counts') 
        display(tab) # display the tab layout
    
        with out1: # display info of the dataframe
            self.info()

        with out2: # create countplots for categorical variables
            fig2 = self.countPlots(splitTarg=True, show=False, optional=optional_countplots)
            plt.show(fig2)
        
        with out3: # create histograms for numerical variables
            fig3 = self.histPlot(kde=True, show=False, optional=optional_histplots)
            plt.show(fig3)

        with out4: # display number of missing values before handling
            print("Missing values before handling :", self.data.isnull().sum().sum()) # before handling
            self.handleMissing()
            print("Missing values after handling :", self.data.isnull().sum().sum()) # after handling
            self

        with out5: # display summary statistics for numerical variables
            display(self.describe())

        with out6: # display potential outliers in the dataframe
            self.displayOutliers(k)

        with out7: # create a correlation matrix for the dataframe
            self.correlationMatrix()

        with out8: # create a histogram for the target variable
            fig8 = self.targetDistribution(show=False)
            plt.show(fig8)

        with out9: # create scatterplots for the relationship between the target variable and each feature
            fig9 = self.featureVsTarget(scatterplot=scatterplot, show=False)
            if scatterplot:
                plt.show(fig9) 

        with out10: # display value counts for categorical variables
            if len(self.cat) > 0:
                @interact(column=self.cat)
                def show_value_counts(column=self.cat):
                    return self.valueCounts([column])
                show_value_counts()
