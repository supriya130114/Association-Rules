# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:44:00 2022

@author: ankush
"""

# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

movie= pd.read_csv ("C:/Users/ankush/Desktop/DataSets/Association dataset/my_movies.csv",encoding='UTF-8') 

movies=movie.iloc[:,0:5]
movies.isna().sum()

# Mode Imputer
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

movies["V3"] = pd.DataFrame(mode_imputer.fit_transform(movies[["V3"]]))
movies["V4"] = pd.DataFrame(mode_imputer.fit_transform(movies[["V4"]]))
movies["V5"] = pd.DataFrame(mode_imputer.fit_transform(movies[["V5"]]))
movies.isnull().sum()  


movies1=movies.values.tolist()



movies_list = []

for i in movies1:
     
    movies_list.append(i)
    

all_movies_list = [i for item in movies_list for i in item]

from collections import Counter # ,OrderedDict
item_frequencies = Counter(all_movies_list)
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:10], x = list(range(0, 10)))
plt.xticks(list(range(0, 10)), items[0:10])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
movies_series = pd.DataFrame(pd.Series(movies_list))


movies_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = movies_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 10)), height = frequent_itemsets.support[0:10])
plt.xticks(list(range(0, 10)), frequent_itemsets.itemsets[0:10], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

###############remove profusion of rules##############
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
final=rules_no_redudancy.sort_values('lift', ascending = False)
