
# Share value between known proportions 

import pandas as pd 
from pandas import read_csv

# because we predict it, we set it in brut here
years_dico = {
  "2018":[661419224.8345854, 675036048.0680673, 696827512.8226824,
   720616329.9876039, 642872916.4634159, 698484095.7726285,
    760543764.5917286, 462595148.81983966, 645758381.6289686,
     779581055.3234987, 688185289.401982, 827735896.4067041,],
  "2019":[694601901.022979, 708634889.2523305, 729623279.6722513,
   752197471.901308, 675293794.5582561, 730714708.4966987,
    792656914.4871238, 495472509.71465117, 678720044.5088043,
     812728809.2628393, 721823470.0304455, 861483781.4517369,],
  "2020":[728543032.1870743, 742837307.8993889, 763882449.2268107,
   786572747.9082036, 709783275.5587082, 765217339.8709661,
    827209017.0992612, 530059521.47053707, 713298593.4973142,
     847320770.222331, 756416851.7348497, 896063078.7782608,]
  }

# :)
dict_repart = {"_11_1": 0,"_11_2": 0,"_11_3": 0,"_11_4": 0,"_11_5": 0,"_11_6": 0,"_11_7": 0,"_11_8": 0,"_11_9": 0,
"_24_1": 0,"_24_2": 0,"_24_3": 0,"_24_4": 0,"_24_5": 0,"_24_6": 0,"_24_7": 0,"_24_8": 0,"_24_9": 0,"_27_1": 0,
"_27_2": 0,"_27_3": 0,"_27_4": 0,"_27_5": 0,"_27_6": 0,"_27_7": 0,"_27_8": 0,"_27_9": 0,"_28_1": 0,"_28_2": 0,
"_28_3": 0,"_28_4": 0,"_28_5": 0,"_28_6": 0,"_28_7": 0,"_28_8": 0,"_28_9": 0,"_32_1": 0,"_32_2": 0,"_32_3": 0,
"_32_4": 0,"_32_5": 0,"_32_6": 0,"_32_7": 0,"_32_8": 0,"_32_9": 0,"_44_1": 0,"_44_2": 0,"_44_3": 0,"_44_4": 0,
"_44_5": 0,"_44_6": 0,"_44_7": 0,"_44_8": 0,"_44_9": 0,"_52_1": 0,"_52_2": 0,"_52_3": 0,"_52_4": 0,"_52_5": 0,
"_52_6": 0,"_52_7": 0,"_52_8": 0,"_52_9": 0,"_53_1": 0,"_53_2": 0,"_53_3": 0,"_53_4": 0,"_53_5": 0,"_53_6": 0,
"_53_7": 0,"_53_8": 0,"_53_9": 0,"_5_1": 0,"_5_2": 0,"_5_3": 0,"_5_4": 0,"_5_5": 0,"_5_6": 0,"_5_7": 0,
"_5_8": 0,"_5_9": 0,"_75_1": 0,"_75_2": 0,"_75_3": 0,"_75_4": 0,"_75_5": 0,"_75_6": 0,"_75_7": 0,"_75_8": 0,
"_75_9": 0,"_76_1": 0,"_76_2": 0,"_76_3": 0,"_76_4": 0,"_76_5": 0,"_76_6": 0,"_76_7": 0,"_76_8": 0,"_76_9": 0,
"_84_1": 0,"_84_2": 0,"_84_3": 0,"_84_4": 0,"_84_5": 0,"_84_6": 0,"_84_7": 0,"_84_8": 0,"_84_9": 0,"_93_1": 0,
"_93_2": 0,"_93_3": 0,"_93_4": 0,"_93_5": 0,"_93_6": 0,"_93_7": 0,"_93_8": 0,"_93_9": 0,"_99_1": 0,"_99_2": 0,
"_99_3": 0,"_99_4": 0,"_99_5": 0,"_99_6": 0,"_99_7": 0,"_99_8": 0,"_99_9": 0}

# define a cost sharing by default because we don't use multivariate
reparti_dico = {}


def get_reparti(reparti_dict,year,month):
  value = sum(reparti_dict[year][month])
  print(value)
  # create the share dictionarry
  som = 0
  result_dict = dict_repart
  for v,(key,coef) in zip(reparti_dict[year][month],dict_repart.items()):
      r = v / value
      result_dict[key] = r
  return result_dict

def fill_dico_with_previous_coef():
  series = read_csv('results-final.csv', header=0)
  series.fillna(0,inplace=True)
  array = series.values
  array.astype(float)
  i = 1
  p = 9
  tab = []
  for ar in array:
    tab.append(ar)
    if i % 12 == 0:
      if p < 10:
        reparti_dico["200"+str(p)] = tab
      elif p >= 10:
        reparti_dico["20"+str(p)] = tab
      p+=1
      tab = []
    i+=1
  return reparti_dico

def fill_dico_with_previous_years():
  series = read_csv('full.csv', header=0,usecols=[1])
  array = series.values
  array.astype(float)
  array = array.reshape(len(array))

  tab = []
  i = 0
  p = 9
  for l in array:
    tab.append(l)
    i +=1
    if i % 12 == 0:
      if p < 10:
        years_dico["200"+str(p)] = tab
      elif p >= 10:
        years_dico["20"+str(p)] = tab
      p+=1
      tab = []
  return years_dico

years_dico = fill_dico_with_previous_years()
reparti_dico = fill_dico_with_previous_coef()

years = ["2009","2010","2011","2012","2013","2014",
"2015","2016","2017"]
#,"2018","2019","2020"]
final_csv = {"Date":[],"Region":[],"Maladie":[],"Depassement":[]}

for year in years:
  month = 1
  for v in years_dico[year]:
      result_dict = get_reparti(reparti_dico,year,month-1)
      print(result_dict)
      for key,value in result_dict.items():
        if len(key) > 4:
          region = key[1]+key[2]
          maladie = key[4]
        else:
          region = key[1]
          maladie = key[3]
        if month < 10:
          date = str(year)+"-0"+str(month)+"-01"
        else:
          date = str(year)+"-"+str(month)+"-01"
        depassement = v * value
        final_csv["Date"].append(date)
        final_csv["Region"].append(region)
        final_csv["Maladie"].append(maladie)
        final_csv["Depassement"].append(depassement)
      month+=1

print(final_csv)
# Convert into pandas
#columns = ["Date","Region","Maladie","Depassement"}
final_data = pd.DataFrame.from_dict(final_csv)

# Save pandas into csv
final_data.to_csv("known_ryu.csv",columns=["Date","Region","Maladie","Depassement"])

# pour chaque valeur total de chaque mois
# on multiplie
    
# pour chaque fois

#DUNS is not globals

#The matching is defaulted to a 999 number, 