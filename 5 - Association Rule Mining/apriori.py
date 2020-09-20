# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

products=['corn']

# Visualising the results
result = list(rules)
dataFrameResult = pd.DataFrame(result)
support = dataFrameResult.support
bought = []
recommendation = []
confidence = []
lift = []
for i in range(dataFrameResult.shape[0]):
    singleList = dataFrameResult["ordered_statistics"][i][0]
    bought.append(list(singleList[0]))
    recommendation.append(list(singleList[1]))
    confidence.append(singleList[2])
    lift.append(singleList[3])
data = {"Bought": bought, "Recommendation": recommendation, "Support": support, "Confidence": confidence, "Lift": lift}
finalDataFrame = pd.DataFrame(data)
finalDataFrame = finalDataFrame.sort_values(by ='Lift', ascending=False)
finalDataFrame.reset_index(inplace = True) 
rec=[]

for i in finalDataFrame.index:
    check_buy = set(finalDataFrame['Bought'][i])
    buy = set(products)
    if (check_buy.issubset(buy)):
        for w in finalDataFrame['Recommendation'][i]:
            rec.append(w)
    if (buy.issubset(check_buy)):
        temp = list(check_buy.difference(buy))
        for w in temp:
            rec.append(w)
    print(len(rec))
    if (len(rec)>5):
        break

print(len(dataset))
finalRec = list(set(rec))
