#------------------------------------------------------------------------------
#                           1. Data Mapping
#------------------------------------------------------------------------------

#Import Pandas Liabrary

import pandas as pd


#Read the file & get duplicate dataset

Prdata=pd.read_csv('project_Data.csv')

PData=Prdata.copy()




#------------------------------------------------------------------------------
#                          2. Data Cleaning
#------------------------------------------------------------------------------

#-----------------------------------
# 2.A  Clean Noisy Data
#-----------------------------------
#Drop first column
PData=PData.drop(PData.columns[0], axis=1)

#Drop id column
PData=PData.drop(PData.columns[0], axis=1)

#check for null values
PData.isnull().sum(axis=0)

#impute null values

#calculate mean value of column 'Arrival Delay in Minutes'
mean_value = int(PData['Arrival Delay in Minutes'].mean())

# Replace null values with the mean value
PData['Arrival Delay in Minutes'].fillna(mean_value, inplace=True)

# Droping rows of null values from other columns as only 1 null value each column has
PData=PData.dropna()

PData.dtypes
#-----------------------------------
# 2.B Check for outliers 
#------------------------------------

import matplotlib.pyplot as plt

plt.boxplot(PData['Age'])   #No outlier

plt.boxplot(PData['Flight Distance'])  #outliers identified 1

plt.boxplot(PData['Inflight wifi service']) #No outlier

plt.boxplot(PData['Departure/Arrival time convenient']) #No outlier

plt.boxplot(PData['Ease of Online booking'])  #No Outlier

plt.boxplot(PData['Gate location'])     # No Outlier

plt.boxplot(PData['Food and drink'])     # No Outlier

plt.boxplot(PData['Online boarding'])     # No Outlier

plt.boxplot(PData['Seat comfort'])        # No Outlier

plt.boxplot(PData['Inflight entertainment'])  # No Outlier

plt.boxplot(PData['On-board service'])      # No Outlier

plt.boxplot(PData['Leg room service'])     # No Outlier

plt.boxplot(PData['Baggage handling'])     # No Outlier

plt.boxplot(PData['Checkin service'])      # Lower Outliers Identified 2

plt.boxplot(PData['Inflight service'])     # No Outlier

plt.boxplot(PData['Cleanliness'])         # No Outlier

plt.boxplot(PData['Departure Delay in Minutes'])  # Outlier Identified 3

plt.boxplot(PData['Arrival Delay in Minutes'])  # Outlier Identified 4




#----------------------------------------
#   2.C Outliers Treatment & Transformation
#----------------------------------------

# 1. Outlier Treatment for 'Flight Distance'


# Calculate quartiles and interquartile range (IQR)
A1 = PData['Flight Distance'].quantile(0.25)
A3 = PData['Flight Distance'].quantile(0.75)
IQR1 = A3 - A1

# Calculate lower and upper bounds for outliers
lower_bound1 = A1 - 1.5 * IQR1
upper_bound1 = A3 + 1.5 * IQR1

# Count outliers
outliers1 = PData['Flight Distance'][(PData['Flight Distance'] < lower_bound1) |\
            (PData['Flight Distance'] > upper_bound1)]
num_outliers1 = len(outliers1)


# Outlier Treatment: Replacing outlier values with mean value

# Filter out rows with outlier values
filtered_df = PData[( PData['Flight Distance'] >=\
                        lower_bound1) & ( PData['Flight Distance'] <= upper_bound1)]

# Replace data within the original DataFrame
PData = filtered_df


# Updated Box plot
plt.boxplot(filtered_df['Flight Distance']) 

"""Few values are still there, maybe values just just above the upper bound.
i.e. : 3739.0"""



    
# 2. Outlier Treatment for 'Checkin service'

# Calculate quartiles and interquartile range (IQR)
B1 = PData['Checkin service'].quantile(0.25)
B3 = PData['Checkin service'].quantile(0.75)
IQR2 = B3 - B1

# Calculate lower and upper bounds for outliers
lower_bound2 = B1 - 1.5 * IQR2
upper_bound2 = B3 + 1.5 * IQR2

# Count outliers
outliers2 = PData['Checkin service'][(PData['Checkin service'] < lower_bound2) |\
            (PData['Checkin service'] > upper_bound2)]
num_outliers2 = len(outliers2)

"""The number of outliers compared to actual data is about 12%.
Which is slightly high in numbers & need to be standerdise."""

# Outlier Treatment: Replacing outlier values with mean value

mean_value2 = PData['Checkin service'].mean()

PData.loc[outliers2.index, 'Checkin service'] = mean_value2


# Updated Box Plot
plt.boxplot(PData['Checkin service'])  




# 3. Outlier Treatment for 'Departure Delay in Minutes'

# Calculate quartiles and interquartile range (IQR)
C1 = PData['Departure Delay in Minutes'].quantile(0.25)
C3 = PData['Departure Delay in Minutes'].quantile(0.75)
IQR3 = C3 - C1

# Calculate lower and upper bounds for outliers
lower_bound3 = C1 - 1.5 * IQR3
upper_bound3 = C3 + 1.5 * IQR3

# Count outliers
outliers3 = PData['Departure Delay in Minutes']\
    [(PData['Departure Delay in Minutes'] < lower_bound3) |\
            (PData['Departure Delay in Minutes'] > upper_bound3)]

num_outliers3 = len(outliers3)  


# Outlier Treatment: Replacing outlier values with mean value

mean_value3 = PData['Departure Delay in Minutes'].mean()

PData.loc[outliers3.index, 'Departure Delay in Minutes'] = mean_value3

# Updated Box Plot
plt.boxplot(PData['Departure Delay in Minutes'])




# 4. Outlier Treatment for 'Arrival Delay in Minutes'

# Calculate quartiles and interquartile range (IQR)
D1 = PData['Arrival Delay in Minutes'].quantile(0.25)
D3 = PData['Arrival Delay in Minutes'].quantile(0.75)
IQR4 = D3 - D1

# Calculate lower and upper bounds for outliers
lower_bound4 = D1 - 1.5 * IQR4
upper_bound4 = D3 + 1.5 * IQR4

# Count outliers
outliers4 = PData['Arrival Delay in Minutes']\
    [(PData['Arrival Delay in Minutes'] < lower_bound4) |\
            (PData['Arrival Delay in Minutes'] > upper_bound4)]

num_outliers4 = len(outliers4)  
    
    
# Outlier Treatment: Replacing outlier values with mean value

mean_value4 = PData['Arrival Delay in Minutes'].mean()

PData.loc[outliers4.index, 'Arrival Delay in Minutes'] = mean_value4
    
# Updated Box Plot
plt.boxplot(PData['Arrival Delay in Minutes'])




#------------------------------------------------------------------------------
#   3. One Hot Encoding to categorical / Object Data Type Columns
#------------------------------------------------------------------------------


PData= pd.get_dummies(PData, drop_first=True)




#------------------------------------------------------------------------------
#   4.  Data Preparation
#------------------------------------------------------------------------------

# Split the columns into Dependent (Y) and independent (X) features
x = PData.iloc[:,:-1]
y = PData.iloc[:, -1]






#------------------------------------------------------------------------------
#                      5. Data Normalisation
#------------------------------------------------------------------------------

# Import the StandardScaler class
from sklearn.preprocessing import StandardScaler

# Create an object of the class StandardScaler
scaler = StandardScaler()

# Fit and Transform the data for normalization
x_scaled = scaler.fit_transform(x)





#------------------------------------------------------------------------------
#                       6. Train Test Split
#------------------------------------------------------------------------------


# Split the X and Y dataset into training and testing set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
train_test_split(x_scaled, y, test_size = 0.3, random_state = 1234, stratify=y)




#------------------------------------------------------------------------------
#                       7. Feature Selection 
#------------------------------------------------------------------------------

#                     F Test (F Score & P Values)


# import and perform the f_regression to get the F-Score and P-Values
from sklearn.feature_selection import f_classif as fr
result = fr(x_scaled,y)


# Split the result tuple into F_Score and P_Values
f_score = result[0]
p_values = result[1]


# Print the table of Features, F-Score and P-values
columns = list(x.columns)

print (" ")
print (" ")
print (" ")

print ("    Features     ","                  F-Score         ",      "P-Values")
print ("    -----------                   ---------                ---------")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("    ", columns[i].ljust(34), f1.rjust(8),"    ", p1.rjust(8))







#             Import and train Random Forest Classifier
#   ---------------------------------------------------------------------



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)
rfc.fit(x_train, y_train)


# Test the RFC model

from sklearn.metrics import confusion_matrix

Y_pred_rfc = rfc.predict(x_test)

# Evaluate the RFC model

cm_rfc = confusion_matrix(y_test, Y_pred_rfc)

Accuracy_rfc = rfc.score(x_test, y_test)





#                   Apply Recursive Feature Elimination
#             -----------------------------------------------

from sklearn.feature_selection import RFE
rfc1 = RandomForestClassifier(random_state=1234)

# Create an RFE selector object using RFC as an estimator
rfe = RFE(estimator=rfc1, n_features_to_select=10, step=1)

# Fit the data to the rfe selector
rfe.fit(x_scaled, y)

# Create new Train and Test datasets
x_train_rfe = rfe.transform(x_train)
x_test_rfe = rfe.transform(x_test)



# Fit the data to the rfe selector
rfc1.fit(x_train_rfe, y_train)

# Test the model with new Test dataset
y_pred_rfc1 = rfc1.predict(x_test_rfe)



# Score and Evaluate the new model 
from sklearn.metrics import confusion_matrix
cm_rfc1 = confusion_matrix(y_test, y_pred_rfc1)
Accuracy_rfc1 = rfc1.score(x_test_rfe, y_test)


"""After using RFE accuracy is going down & When less features elimnated it gives 
almost same accuracy"""
"""So RFE is not recomended here"""





#                  Use PCA (Principal Component Analysis)
#          --------------------------------------------------------



# Check the mean of the centered data
x_mean=x_scaled[:,0].mean()


# Import PCA and fit the data to create PCAs
from sklearn.decomposition import PCA
pca = PCA(n_components=18)
x_pca_rfc = pca.fit_transform(x_scaled)



# Split the dataset into train and test
x_train, x_test, y_train, y_test = \
train_test_split(x_pca_rfc, y, test_size = 0.3, random_state = 1234, stratify=y)



# Default RFC Object

rfc2 = RandomForestClassifier(random_state=1234)
rfc2.fit(x_train, y_train)
y_pred_rfc2 =rfc2.predict(x_test)


# Score and Evaluate the model using transformed data

cm_rfc2 = confusion_matrix(y_test, y_pred_rfc2)
Accuracy_rfc2 = rfc2.score(x_test, y_test)


"""After applying PCA if we eliminate few input features using PCA, 
the accuracy of the model decreases. Hence all the input features shows strong 
influence on target variable"""






#    Import various select transforms along with the f_classif mode
#  -----------------------------------------------------------------------


from sklearn.feature_selection import SelectKBest,             \
                                      SelectPercentile,        \
                                      GenericUnivariateSelect, \
                                      f_classif






#                    Implement SelectPercentile
#           ------------------------------------------------

selectorP = SelectPercentile(score_func=f_classif, percentile=50)
x_percentile = selectorP.fit_transform(x_scaled, y)


# Split the dataset into train and test
x_train, x_test, y_train, y_test = \
train_test_split(x_percentile, y, test_size = 0.3, random_state = 1234, stratify=y)

# Default RFC Object

rfc3 = RandomForestClassifier(random_state=1234)
rfc3.fit(x_train, y_train)
y_pred_rfc3 =rfc3.predict(x_test)


# Score and Evaluate the model using transformed data

cm_rfc3 = confusion_matrix(y_test, y_pred_rfc3)
Accuracy_rfc3 = rfc3.score(x_test, y_test)

"""Again after aplying Select-percentile class accuracy reduces & when we increase
parameter close to 100 accuracy increases. Which shows the select-percentile class is
inversely propotional to the accuracy. Hence it does not add any value to the model.
& We are not selecting the class for RFC model with the given Data"""






#                      Implement SelectKBest
#         --------------------------------------------------


selector_kbest_rfc = SelectKBest(score_func=f_classif, k=13)

x_kbest_rfc = selector_kbest_rfc.fit_transform(x_scaled, y)


# Split the dataset into train and test

x_train, x_test, y_train, y_test = \
train_test_split(x_kbest_rfc, y, test_size = 0.3, random_state = 1234, stratify=y)

# Default RFC Object

rfc4 = RandomForestClassifier(random_state=1234)
rfc4.fit(x_train, y_train)
y_pred_rfc4 =rfc4.predict(x_test)


# Score and Evaluate the model using transformed data

cm_rfc4 = confusion_matrix(y_test, y_pred_rfc4)
Accuracy_rfc4 = rfc4.score(x_test, y_test)


"""In select-kbest class parameter of k is inversely propotional to accuracy.
Hence we can not apply this class for the given data set with RFC model"""






#       --------------------------------------------------------------
#                  Implement Generic Univariate Select
#       --------------------------------------------------------------


#              Implement GenericUnivariateSelect with k_best
#      ----------------------------------------------------------------



selectorG_rfc1 = GenericUnivariateSelect(score_func=f_classif, mode='k_best', param=13)

x_Generic_rfc1 = selectorG_rfc1.fit_transform(x_scaled,y)

# Split the dataset into train and test

x_train, x_test, y_train, y_test = \
train_test_split(x_Generic_rfc1, y, test_size = 0.3, random_state = 1234, stratify=y)


# Default RFC Object

rfc5 = RandomForestClassifier(random_state=1234)
rfc5.fit(x_train, y_train)
y_pred_rfc5 =rfc5.predict(x_test)


# Score and Evaluate the model using transformed data

cm_rfc5 = confusion_matrix(y_test, y_pred_rfc5)
Accuracy_rfc5 = rfc5.score(x_test, y_test)



"""In GenericUnivariateSelect class with k-best mode is inversely propotional to
 accuracy.Hence we can not apply this class for the given data set with RFC model"""



#        Implement GenericUnivariateSelect with Percentile
#   -----------------------------------------------------------

# Implement GenericUnivariateSelect with percentile
SelectorG_rfc2 = GenericUnivariateSelect(score_func=f_classif,
                                     mode='percentile',
                                     param=50)

x_Generic_rfc2 = SelectorG_rfc2.fit_transform(x_scaled, y)


# Split the dataset into train and test

x_train, x_test, y_train, y_test = \
train_test_split(x_Generic_rfc2, y, test_size = 0.3, random_state = 1234, stratify=y)


# Default RFC Object

rfc6 = RandomForestClassifier(random_state=1234)
rfc6.fit(x_train, y_train)
y_pred_rfc6 =rfc6.predict(x_test)


# Score and Evaluate the model using transformed data

cm_rfc6 = confusion_matrix(y_test, y_pred_rfc6)
Accuracy_rfc6 = rfc6.score(x_test, y_test)


"""In GenericUnivariateSelect class with percentile parameter is inversely propotional to
accuracy.Hence we can not apply this class for the given data set with RFC model"""




"""P S: In Random Forest Classifier RFE,PCA, Select K-Best, Select Percentile,
Generic Univariate with f_classif dosent do any significant changes.
In fact after applying them the accuracy decreasess"""
















