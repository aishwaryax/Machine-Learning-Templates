import csv

with open('Market_Basket_Optimisation.csv', 'r') as f:
  file = csv.reader(f)
  my_list = list(file)

products = []

for i in my_list:
    for j in i:
        products.append(j)

productsSet = list(set(products))

from collections import defaultdict

food_count = defaultdict(int) # default value of int is 0


for w in products:
    food_count[w]=food_count[w]+1

res = sorted(food_count.items(), key=lambda x: x[1], reverse=True)
result = []
for w in res:
    result.append(w[0])
    if (len(result)>=5):
        break
    
    