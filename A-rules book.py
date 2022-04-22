# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:54:44 2022

@author: ankush
"""

# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

book= pd.read_csv ("C:/Users/ankush/Desktop/DataSets/Association dataset/book.csv") 

from collections import Counter # ,OrderedDict

item_frequencies = Counter(book)
book.columns
# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(([i[1] for i in item_frequencies]))
items = list(([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'red')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

frequent_itemsets = apriori(book, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='red')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
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
b=rules_no_redudancy.sort_values('lift', ascending = False).head(10)
allr=rules_no_redudancy.sort_values('lift', ascending = False)
