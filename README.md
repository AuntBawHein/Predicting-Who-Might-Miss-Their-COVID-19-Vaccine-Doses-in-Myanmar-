# Predicting-Who-Might-Miss-Their-COVID-19-Vaccine-Doses-in-Myanmar-

Used Python to create a program that predicts which people in Myanmar might miss their COVID-19 vaccine doses. The program looks at things like age, gender, and previous vaccine records to help doctors know who needs extra reminders or care.

Step 1: Define Scope/Objectives

•	Objective: I'm going to clearly say what the goal of our project is and what we want to achieve.

•	Tasks:

•	I'm going to figure out the main questions we need to answer. For example, we want to find out if someone might miss their second or third COVID-19 vaccine based on their age, where they live, and if they've missed doses before.

• I'm going to think about who will use our findings, like doctors or government officials, and how they can use this information to help people get vaccinated.

•	I'm going to decide what we will create at the end, like a tool that predicts who might miss their vaccine, some graphs, and a report that explains what we found.

What do you want to achieve or do?

= We want to figure out who might miss their second or third COVID-19 vaccine dose by using data about their age, gender, and vaccine history.

Why do this project?

= This project is important because it helps make sure people get all their vaccine doses, keeping them and the community safe from the virus.

Step 2: Data Loading and Initial Exploration

In this step, I'm going to load our data and start exploring it to see what it looks like.

•	Objective: I'll load the data we need and start figuring out what's in it.

•	Tasks:

• I'm going to use some tools like pandas, numpy, matplotlib, and seaborn to help us handle and visualize the data.

• I'll load the data into our project using pandas and look at the first few rows to see how the information is organized.

•	I'll use .info() to check what types of data we have in each column and to see if any important information is missing.

•	I'll use .describe() to get basic statistics that tell us more about numbers in our data, like the ages of people in the study.

________________________________________

What is Pandas?

Pandas is a tool that helps us easily work with data. It lets us clean, duplicate, and manipulate data.

Why do we have to use Pandas in this step?

I'm going to use pandas to load our data into the project so we can start working with it. It makes it easy to look at and understand the data.

________________________________________

What is Numpy?

Numpy is a tool that helps us do math and work with numbers in our data. It’s really fast and powerful for handling large amounts of data.

Why do we have to use Numpy in this step?

I'm going to use numpy to help with calculations and to make it easier to manage the data we load. It works well with pandas to make our data easier to work with.

________________________________________

What is Matplotlib?

Matplotlib is a tool that helps us create charts and graphs. It turns numbers into pictures, so we can see patterns in the data.

Why do we have to use Matplotlib in this step?

I'm going to use matplotlib to make basic charts and graphs. This helps us see what’s going on in the data, like how many people are in each age group.

________________________________________

What is Seaborn?

Seaborn is a tool that makes our charts and graphs look even better. It builds on matplotlib and makes it easier to create more advanced visualizations.

Why do we have to use Seaborn in this step?

I'm going to use seaborn to make our graphs look nicer and to help us spot trends in the data more clearly.

Code

# Step 2: Data Loading and Initial Exploration

import pandas as pd  # I import pandas to handle data tables easily.

import matplotlib.pyplot as plt  # I import matplotlib to create clear graphs.

# I load the data from the CSV file into a table.

data = pd.read_csv("Covid_vaccination_record_refugee_camp.csv")

# I look at the first few rows to see what the data looks like.

print(data.head())

# I create a histogram to clearly show how the ages are spread out.

plt.figure(figsize=(8, 6))  # I set the size of my chart clearly.

plt.hist(data['Age'], bins=10, color='skyblue', edgecolor='black') # I create the histogram clearly.

plt.xlabel("Age (Years)")  # I label the bottom as "Age (Years)".

plt.ylabel("Number of People")  # I label the side clearly to show counts.

plt.title("Age Distribution of People in the Study")  # I give a clear title.

plt.grid(True)  # I add grid lines for clarity.

plt.show()  # I show the chart.

What I’m going to do:

[Covid_vaccination_record_refugee_camp.csv](https://github.com/user-attachments/files/19460286/Covid_vaccination_record_refugee_camp.csv)

[Covid vaccination record in a refugee camp.xlsx](https://github.com/user-attachments/files/19460285/Covid.vaccination.record.in.a.refugee.camp.xlsx)

![image](https://github.com/user-attachments/assets/877291aa-c3f5-4d13-94f4-c65a58c13822)

Conclusions:

I found that most people in the Myanmar refugee camp who got vaccinated are young, especially under 20 years old.  

I also noticed that fewer older people, like those over 60, received vaccines compared to younger people.

Step 3: Data Cleaning and Preprocessing

In this step, I'm going to clean up the data to make sure it's ready for analysis and to build a prediction model.

•	Objective: I will clean the dataset so it’s ready for us to analyze and use to make predictions.

•	Tasks:

•	I'm going to handle any missing values. If we can guess what the missing value should be, I’ll fill it in. If there are too many missing values in a row or column, I might remove them.

•	I'm going to convert the date columns (First Dose Date, Second Dose Date, Third Dose Date, Vaccine Record Card Date) into a date format that the computer can easily understand.

•	I'm going to create new information from the data, like calculating how much time passed between each vaccine dose.

•	I'm going to change words in the data (like Sex and Vaccine Type) into numbers, because the computer can work with numbers better when making predictions. This process is called encoding.

________________________________________

What is Pandas?

Pandas is a tool that helps us easily work with data in tables. It lets us clean, manu data in a way that is easy to understand and use.

Why do we have to use Pandas in this step?

I'm going to use Pandas to load our data, clean it up, and get it ready for analysis. Pandas makes it easy to handle missing values, convert dates, and organize the data into a format that we can use for making predictions.

________________________________________

What is Numpy?

Numpy is a tool that helps us do math with data. It's very fast and powerful for handling numbers, especially when we have large amounts of data.

Why do we have to use Numpy in this step?

I'm going to use Numpy to help with calculations, like finding the average value of numbers or working with large datasets. Numpy works well with Pandas to make data processing faster and easier.

Code

# Step 3: Data Cleaning and Preprocessing

import pandas as pd  # I import pandas to handle data tables easily

import numpy as np  # I import numpy to do math calculations easily

# Load the dataset

data = pd.read_csv("Covid_vaccination_record_refugee_camp.csv")

# Handle missing values in 'Age' by filling with average age

if 'Age' in data.columns:

    data['Age'] = data['Age'].fillna(data['Age'].mean())

# Drop 'Third Dose Date' column if too many values are missing

if data['Third Dose Date'].isnull().sum() > len(data) * 0.5:

    data = data.drop(columns=['Third Dose Date'])

# Convert date columns to a date format the computer can easily understand

date_columns = ['First Dose Date', 'Second Dose Date', 'Vaccine Record Card Date']

for column in date_columns:

    if column in data.columns:
    
        data[column] = pd.to_datetime(data[column], errors='coerce')

# Create a new feature: days between first and second doses

if 'First Dose Date' in data.columns and 'Second Dose Date' in data.columns:

    data['Time Between Doses'] = (data['Second Dose Date'] - data['First Dose Date']).dt.days

# Turn categorical data (like Sex and Vaccine Type) into numbers

if 'Sex' in data.columns:

    data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

if 'Vaccine Type' in data.columns:

    data = pd.get_dummies(data, columns=['Vaccine Type'], drop_first=True)

# Checking cleaned data

print("Cleaned data preview:")

print(data.head())

Missing values in each column before cleaning:

Sr.                             0

Address                         2

Section                         2

House no.                       4

Date of Birth                 399

Age                             3

Sex                             0

AG                              0

First Dose Vaccine          19572

First Dose Date             19573

Second Dose Vaccine         22436

Second Dose Date            22438

Third Dose Vaccine          30363

Third Dose Date             30366

Vaccine Record Card Date    23024

Missing Second Dose             0

dtype: int64

First 5 rows of the cleaned and processed dataset:

   Sr. Address Section House no.        Date of Birth   Age            AG  \
   
0    1    A1/1      A1         1  1975-06-07 00:00:00  46.0  18 and Above 

1    2    A1/2      A1         2  2001-08-06 00:00:00  20.0  18 and Above 

2    3    A1/2      A1         2  1995-07-05 00:00:00  26.0  18 and Above 

3    4    A1/2      A1         2  1974-05-12 00:00:00  47.0  18 and Above 

4    5    A1/2      A1         2  1973-02-06 00:00:00  48.0  18 and Above   

  First Dose Vaccine First Dose Date Second Dose Vaccine Second Dose Date  \
  
0            Sinovac      2021-10-29         AstraZeneca       2021-11-23 

1                NaN             NaT                 NaN              NaT 

2                NaN             NaT                 NaN              NaT 

3                NaN             NaT                 NaN              NaT 

4                NaN             NaT                 NaN              NaT   

  Third Dose Vaccine Vaccine Record Card Date  Missing Second Dose  \
  
0             Pfizer               2022-03-08                    0 

1                NaN                      NaT                    1 

2                NaN                      NaT                    1 

3                NaN                      NaT                    1 

4                NaN                      NaT                    1   

   Time Between Doses  Sex_Male  
   
0                25.0      True  

1                 NaN      True 

2                 NaN      True 

3                 NaN     False 

4                 NaN      True  

Missing values in each column after cleaning:

Sr.                             0

Address                         2

Section                         2

House no.                       4

Date of Birth                 399

Age                             0

AG                              0

First Dose Vaccine          19572

First Dose Date             19573

Second Dose Vaccine         22436

Second Dose Date            22438

Third Dose Vaccine          30363

Vaccine Record Card Date    23029

Missing Second Dose             0

Time Between Doses          22443

Sex_Male                        0

dtype: int64

Conclusions:

I noticed that after cleaning the data, the "Age" column had no missing values left (0 missing values). 

However, I still saw many missing values in "Third Dose Vaccine," with 30,363 missing entries, which is much higher than other columns.

Step 4: Exploratory Data Analysis (EDA)

In this step, I'm going to explore the data using visual tools to find patterns, trends, and relationships.

•	Objective: I'm going to explore the dataset visually to see if there are any interesting patterns or connections.

•	Tasks:

•	I'm going to use tools like Seaborn and Matplotlib to create charts and graphs. I'll make histograms to look at how age, sex, and vaccine type are spread out, and bar plots to compare different groups.

•	I'm going to look at how different features (like age and whether someone got all their vaccine doses) are related. This will help us understand how one thing might affect another.

•	I'm going to use heatmaps to show how strongly different numbers (like age, time between doses, etc.) are connected to each other.

•	I'm going to check if there are any outliers or unusual data points that look very different from the rest. These might need more attention or explanation.

________________________________________

What is Matplotlib?

Matplotlib is a library in Python that I’m going to use to make basic charts and graphs, like bar charts and line graphs.

Why do I have to use Matplotlib for this step?

I’m going to use Matplotlib to help me visualize the data. This makes it easier to see patterns, trends, and any unusual points in the data.

________________________________________

What is Seaborn?

Seaborn is another library in Python, similar to Matplotlib, but it helps me create nicer and more detailed charts and graphs.

Why do I have to use Seaborn for this step?

I’m going to use Seaborn because it makes the charts look better and it’s easier to create more complex visualizations like heatmaps.

________________________________________

What is Pandas?

Pandas is a library in Python that I’m going to use to work with the data like a table or a spreadsheet. It helps me organize and analyze the data.

Why do I have to use Pandas for this step?

I’m going to use Pandas to load the data, organize it, and prepare it for making charts and graphs. It makes working with large amounts of data much easier.

Code

# Import important libraries

import matplotlib.pyplot as plt  # I import matplotlib to create clear graphs.

import seaborn as sns  # I import seaborn to make the graphs look nice.


# I use pandas to load the data into my project.

data = pd.read_csv("Covid_vaccination_record_refugee_camp.csv")

# I draw a clear histogram graph to show how the ages are spread out.

plt.figure(figsize=(8, 6))

sns.histplot(data['Age'], bins=10, kde=True)

plt.title("Age Distribution - Myanmar Refugee Camp COVID-19")

plt.xlabel("Age (Years)")

plt.ylabel("Number of Individuals")

plt.show()

# I draw a clear boxplot to find if there are any unusual ages (outliers).

plt.figure(figsize=(8, 6))

sns.boxplot(x=data['Age'])

plt.title("Outliers in Age - Myanmar Refugee Camp COVID-19")

plt.xlabel("Age (Years)")

plt.show()

![image](https://github.com/user-attachments/assets/857671f0-94e3-4af7-8e0b-0dbe64d8b26e)
![image](https://github.com/user-attachments/assets/4569e950-5e7b-4e87-9167-23d6c7ffa691)
![image](https://github.com/user-attachments/assets/e239dd8e-0463-4414-a667-347ca6c9189c)
![image](https://github.com/user-attachments/assets/2aa05918-ee3d-441b-8a25-0a9e49d1eb7b)
![image](https://github.com/user-attachments/assets/4f2b1653-1a7a-46b2-a019-8012e4099f0f)
![image](https://github.com/user-attachments/assets/d21aea81-928a-44eb-b3ec-a3e97bc1efab)

Conclusions:

1. I noticed that most people vaccinated in the Myanmar refugee camp are younger, around 20 years old.

2. I saw that both males and females got vaccinated almost equally, meaning there was no big difference in gender distribution.

3. I found that Sinovac was used less often compared to Pfizer and Sinopharm vaccines.

4. I discovered that the time between the first and second doses was usually around 50 days, regardless of age.

5. I observed a small relationship between age and time between doses; older people sometimes waited longer between doses.

6. I saw some people had unusual waiting times, like negative days or more than 300 days, which might be mistakes or special cases.

7. I found that the age and time between doses were not strongly connected (for example, a low connection like 0.069), meaning age doesn't always affect how long someone waits between doses.

Step 5: Feature Engineering

In this step, I'm going to make the data more useful by creating new information from the existing data.

•	Objective: Make the dataset better by creating new pieces of information that can help our prediction work better.

•	Tasks:

• I'm going to create a new feature called Age Group to sort people into groups like 18-30 years old, 31-45 years old, etc.

•	I'm going to calculate how many days there are between the first and second vaccine doses, and also between the second and third doses.

•	I'm going to create new columns that show if someone is missing a dose (for example, a column that shows if the second dose is missing).

•	I'm going to change text data into numbers and make sure all the numbers are on the same scale, so the data is ready for making predictions.
________________________________________

What is Pandas?

Pandas is a tool that helps us work with data in a table format, like a spreadsheet. It makes it easy to organize, analyze, and manipulate data.

Why do we have to use Pandas for this step?

I'm going to use Pandas to handle the data table. It allows me to create new columns, change data formats, and calculate new information like age groups and time between doses.

________________________________________

What is from sklearn.preprocessing import MinMaxScaler?

MinMaxScaler is a tool from a library called sklearn. It helps us change numbers so that they all fit within a certain range, like from 0 to 1. This process is called scaling.

Why do we have to use from sklearn.preprocessing import MinMaxScaler for this step?

I'm going to use MinMaxScaler to make sure all the numbers in the data are on the same scale. This is important because it helps the computer understand the data better when making predictions.

Code

# Create Age Group feature

bins = [0, 17, 30, 45, 60, 75, 100]

labels = ['0-17', '18-30', '31-45', '46-60', '61-75', '76-100']

data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Calculate time between first and second doses

data['Time Between First and Second Dose'] = (data['Second Dose Date'] - data['First Dose Date']).dt.days

# Show if second dose is missing

data['Missing Second Dose'] = data['Second Dose Date'].isnull().astype(int)

Step 6: Model Building

In this step, I'm going to create a computer program that can guess if someone might not get all their vaccine doses.

•	Objective: I'm going to build a tool that can predict if someone might miss their vaccine appointments.

•	Tasks:

o	I'm going to split the information into two parts: one part for teaching the tool and one part for testing how well it learned.

o	I'm going to pick a type of tool, like Logistic Regression or Decision Tree, which can help make these guesses.

o	I'm going to teach the tool with the first part of the information and then see how well it does using scores like accuracy, precision, and recall.

o	I'm going to try different settings on the tool to see if it can make better guesses.

________________________________________

What Are These Tools?

What is Pandas?

Pandas is a tool that helps me work with data, like organizing information into tables with rows and columns, just like in Excel.

What is from sklearn.model_selection import train_test_split?

This tool helps me divide my data into two parts: one part to teach the model how to make predictions, and another part to test how well the model learned.

What is from sklearn.linear_model import LogisticRegression?

This is the tool I’m going to use to make the actual predictions. It helps decide if someone might miss their vaccine doses based on the data.

What is from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score?

These are the tools that help me figure out how good my predictions are. They tell me if the model is guessing correctly and how often it catches the right answers.

What is from sklearn.impute import SimpleImputer?

This tool helps me fill in any missing information in the data. For example, if someone's age or address is missing, this tool helps guess what that might be.

________________________________________

Why Do I Need These Tools?

Why do I have to use Pandas for this step?

I’m going to use Pandas because it makes it easy to organize and look at my data, which is the first step before I start making predictions.

Why do I have to use from sklearn.model_selection import train_test_split for this step?

I’m going to use this because I need to teach the model with one part of the data and check how well it learned with another part. This helps make sure my predictions are accurate.

Why do I have to use from sklearn.linear_model import LogisticRegression for this step?

I’m going to use this tool because it’s good at predicting things when we have information like yes/no questions, like whether someone will miss their vaccine doses.

Why do I have to use from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score?

I’m going to use these tools because they help me measure how good my predictions are, so I can understand if my model is working well or if it needs improvement.

Why do I have to use from sklearn.impute import SimpleImputer?

I’m going to use this because sometimes data is missing, and I need to fill in the gaps to make sure my model has all the information it needs to make accurate predictions.

Code

# Select important features and target

X = data_sample[['Age']]  # You can add more features that exist in your dataset

y = data_sample['Missing Second Dose']  # This is what we want to predict

# Split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)

model.fit(X_train, y_train)

# Evaluate the model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

Accuracy: 0.65

Precision: 0.67

Recall: 0.91

F1 Score: 0.77

Conclusions: 

I tested my model and found that it was correct 65% of the time.

I noticed that when my model said someone would miss their vaccine, it was right 67% of the time.

I found that it caught 91% of the people who actually missed their second dose.

I learned that my model had a good balance, with an F1 score of 0.77, which means it worked quite well.

Step 7: Model Evaluation and Validation

In this step, I'm going to check if my computer program (the model) is good at predicting correctly and can be trusted with new information.

•	Objective: Make sure the model is really good at its job and can handle new situations well.

•	Tasks:

o	I’m going to use cross-validation to test the model with different parts of our data to make sure it's consistent.

o	I’m going to create a confusion matrix, which is like a report card, to see when the model is getting things right or wrong.

o	I’m going to draw ROC curves and calculate the AUC score to check how sharp our model is at spotting differences between people who might miss their vaccine doses and those who won't.

o	I’m going to compare this model to others to find out which one does the best job.

This step helps us trust that our model will do well in real-life situations, not just in our tests.

________________________________________

What is Pandas?

Pandas is a tool in Python that helps us work with big tables of data easily. We use it to read, organize, and clean our data.

What is from sklearn.model_selection import train_test_split, cross_val_score?

This command brings in tools from a library called sklearn that help us split our data into parts for training and testing, and to check if our model is doing a good job by testing it many times with different parts of the data.

What is from sklearn.ensemble import RandomForestClassifier?

This command brings in a special type of model called a Random Forest. It’s like using a team of decision trees to make better predictions together.

What is from sklearn.metrics import confusion_matrix, roc_curve, auc?

These tools help us check how well our model is working. The confusion matrix is like a report card, the ROC curve shows how good the model is at distinguishing between different outcomes, and the AUC score tells us how well the model is doing overall.

What is from sklearn.impute import SimpleImputer?

This tool helps us fill in any missing data in our table, so our model doesn’t get confused by empty spots.

What is import matplotlib.pyplot as plt?

This tool helps us make graphs and charts in Python, which we can use to visualize how well our model is working.

________________________________________

Why do we have to use Pandas for this step?

We need Pandas to organize and prepare our data so that we can use it to train and test our model.

Why do we have to use from sklearn.model_selection import train_test_split, cross_val_score for this step?

We use these tools to split our data into training and testing parts, and to check if the model works well with different parts of the data, making sure it's consistent.

Why do we have to use from sklearn.ensemble import RandomForestClassifier for this step?

We use this model because it’s powerful and can make good predictions by combining many decision trees together.

Why do we have to use from sklearn.metrics import confusion_matrix, roc_curve, auc for this step?

We need these tools to see how well our model is performing. They help us understand where the model is doing well and where it might be making mistakes.

Why do we have to use from sklearn.impute import SimpleImputer for this step?

We use this tool to fill in any missing data in our table so that our model can work without any problems.

Why do we have to use import matplotlib.pyplot as plt for this step?

We use this tool to make visual charts and graphs that help us see and understand how well our model is working.

Code 

# Step 6: ROC Curve and AUC

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

# Predict the outcomes

y_pred = model.predict(X_test)

# Calculate ROC curve and AUC

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

# Plotting the ROC curve

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate (Myanmar Medical)")

plt.ylabel("True Positive Rate (Covid-19 Prediction)")

plt.title("ROC Curve - Vaccine Dose Prediction")

plt.legend(loc="lower right")

plt.show()

Cross-validation scores:  [1.         0.99957966 0.99957966]

Confusion Matrix:

 [[1063    0]
 
 [   0 1997]]
 
AUC Score:  1.0

![image](https://github.com/user-attachments/assets/070b2ee2-4eab-4d77-8e6a-d3c71683b556)

Conclusions:

I tested my model and I saw it got an AUC score of 1.0, which means it made perfect predictions.

I looked at the confusion matrix and I found that the model predicted both "missed" and "not missed" vaccine doses correctly — there were no mistakes.

I used cross-validation and I got scores like 1.0, 0.9995, and 0.9995, which shows the model works really well again and again.

I compared this model with others and I noticed this one is more accurate and reliable.

I believe this model can really help health workers in Myanmar find people who may miss their COVID-19 vaccine doses.

Step 8: Visualization

In this step: I'm going to make charts and graphs to help everyone understand the results of our model. This includes showing which factors are most important and how well the model predicts outcomes.

________________________________________

What is Pandas?

Pandas is a tool in Python that helps us work with data in tables.

What is matplotlib.pyplot as plt?

Matplotlib is a tool that helps us make charts and graphs. We use plt to create and show these graphs.

What is Seaborn as sns?

Seaborn is a tool that makes our charts and graphs look nicer and easier to understand.

What is from sklearn.metrics import roc_curve, auc?

This is a tool that helps us check how well our model can predict things by creating a special graph called an ROC curve and calculating a score called AUC.

________________________________________

Why do we have to use Pandas for this step?

I'm going to use Pandas to organize and handle our data in table form so we can easily make charts and graphs.

Why do we have to use matplotlib.pyplot as plt for this step?

I'm going to use plt to create and show the charts and graphs that help us understand the model's results.

Why do we have to use Seaborn as sns for this step?

I'm going to use Seaborn to make the charts and graphs look better and easier to read.

Why do we have to use from sklearn.metrics import roc_curve, auc for this step?

I'm going to use these tools to check how well our model predicts who might miss their vaccine dose by creating an ROC curve and calculating the AUC score.

Code 

# Bar Chart: Missing Second Dose by Age Group

sns.countplot(x='Age Group', hue='Missing Second Dose', data=data)

plt.title("Missing Second Dose by Age Group")

plt.xlabel("Age Group")

plt.ylabel("Number of People")

plt.legend(title='Missing Dose', labels=['No', 'Yes'])

plt.show()

# ROC Curve

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.title("ROC Curve")

plt.show()

# Bar Chart: Top 10 Important Features

top_features = model.feature_importances_.argsort()[-10:][::-1]

sns.barplot(x=model.feature_importances_[top_features], 

            y=data_encoded.columns.drop("Missing Second Dose")[top_features])
            
plt.title("Top 10 Features in Predicting Missed Doses")

plt.xlabel("Importance")

plt.ylabel("Features")

plt.show()

![image](https://github.com/user-attachments/assets/f2b41423-e2da-4af9-8208-92036e0a0721)
![image](https://github.com/user-attachments/assets/8f7a775c-c627-4759-ab6b-fdad2c569b17)
![image](https://github.com/user-attachments/assets/1262f20b-ae69-4981-b67f-4ccc5c5c4351)

Conclusions: 

I found that more young people aged 0–17 missed their second COVID-19 dose compared to older age groups.

I saw that my model was very accurate, with an AUC score of 1.00, meaning it could correctly tell who might miss their dose.

I learned that the type of second dose vaccine, especially those marked as "Other," was the most important factor in the prediction. 

Step 9: Asking and Answering Questions about Predicting Who Might Miss Their COVID-19 Vaccine Doses in Myanmar using Python

 In this step, I'm going to create my own 10 questions and provide simple answers based on real-life scenarios related to the project.
 
________________________________________

1. Why is it important to predict who might miss their COVID-19 vaccine dose?
 
Answer: Predicting who might miss a dose helps healthcare workers take action to ensure that everyone gets fully vaccinated, reducing the risk of COVID-19 spread.

2. How can knowing someone's age help predict missed doses?
   
Answer: Age can be a clue because certain age groups, like younger people, may be more likely to miss their doses. This helps focus efforts on those groups.

3. What did the data reveal about the type of vaccine and missed doses?
 
Answer: The data showed that the type of vaccine someone receives can affect whether they miss a dose, with some vaccines being missed more than others.

4. How does this model make it easier for healthcare workers?
 
Answer: The model identifies people who are at higher risk of missing doses, allowing healthcare workers to target those individuals and encourage them to get vaccinated.

5. Why was a bar chart useful in this analysis?
   
Answer: A bar chart made it easy to see which age groups were missing the most doses, helping us focus on the right groups.

6. What does a perfect ROC curve tell us about our model?
   
Answer: A perfect ROC curve with an AUC of 1.00 means our model is very accurate in predicting who might miss their dose.

7. Which feature was the most important in the model's predictions?

Answer: The type of vaccine received was the most important feature in predicting whether someone would miss a dose.

8. What coding challenge did we face, and how was it solved?
 
Answer: We faced a challenge with missing data, which was solved by filling in the missing values to ensure our model could work properly.

9. How can this project help improve vaccination strategies?

Answer: By identifying who is likely to miss their doses, healthcare teams can create better strategies to ensure everyone gets vaccinated on time.

10. What can be done with the results of this project?

Answer: The results can be used to develop targeted reminders and follow-ups for people at risk of missing their doses, improving overall vaccination rates in the community.

________________________________________
