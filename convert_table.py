# convert group table in to splitted table with 80+ columns

import pandas as pd
import json
from pprint import pprint

with open('results-20181108-202123.json') as f:
    data = json.load(f)
ages = ["age_0","age_20","age_30","age_40","age_50","age_60","age_70","age_80","age_99"]
sexes = ["F_SEX","M_SEX"]
final_df = None
for item in data:
    csv_columns = {"FLX_ANN_MOI":0 , "M_SEX":0, "F_SEX":0 ,"age_0":0,"age_20":0,"age_30":0,
    "age_40":0,"age_50":0,"age_60":0,"age_70":0,"age_80":0,"age_99":0}
    csv_columns["FLX_ANN_MOI"] = item["FLX_ANN_MOI"]
    for age in ages:
        som = 0.0
        for i in item[age]:
            som+= float(i)    
        csv_columns[age]+=som
    for sex in sexes:
        som = 0.0
        for i in item[sex]:
            som+= float(i)
        csv_columns[sex]+=som
    for i in range(len(item["maladie"])):
        n_c = item["reg"][i]+"_"+item["maladie"][i]
        if n_c in csv_columns:
            csv_columns[n_c]+=float(item["dep"][i])
        else:
            csv_columns[n_c]=float(item["dep"][i])
    df = pd.DataFrame(csv_columns,index=[0])
    try:
        if final_df == None:
            final_df = df
        else:
            print("append done IN ELSE")  
            final_df = final_df.append(df,sort=True)
    except TypeError:
        print("append done IN EXCEPT")  
        final_df = final_df.append(df,sort=True)
    #print(final_df)

final_df.to_csv("csv_file_01.csv")
#print(final_df)

# pour chaque couple reg et mnt

# on prends la valeure a partir de l'index 
# reg = "86", mnt = 232, maladie = "12"
# average des sex

'''
pprint(data)
with open("./csv_test.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
'''