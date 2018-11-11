from numpy import array

# Tools functions

#convert forecast sequences to values format
def forecast_to_base_format(len_columns,tab):
    big_tab = []
    p = 1
    for m_tab in tab:
        p = 1
        t_tab = []
        for x in range(len(m_tab)):
            t_tab.append(m_tab[x])
            #print(t_tab)
            #print(t_tab,len_columns*p),
            if x == (len_columns*p)-1:
                #print(t_tab)
                big_tab.append(t_tab)
                p+=1                                                                                                                                                                                                                                                                                                                                
                t_tab = []
    return array(big_tab)

#convert values format to forecast sequences
def base_format_to_forecast(forecast,tab):
    big_tab = []
    t_tab = []
    p = 1
    for m in range(len(tab)):
        for x in tab[m]:
            t_tab.append(x)
        #print(t_tab)
        if m == (forecast*p)-1:
            big_tab.append(t_tab)
            t_tab = []
            p+=1
    return array(big_tab)   

#convert values format to forecast sequences
def base_second_format_to_forecast(forecast,tab):
    big_tab = []
    t_tab = []
    p = 1
    print("forecast : ",forecast)
    print(range(len(tab)))
    for m in range(len(tab)):
        for x in tab[m]:
            t_tab.append(x)
        #print(t_tab)
        if m == (forecast*p)-2:
            print("c'est entre")
            big_tab.append(t_tab)
            t_tab = []
            p+=1
    return array(big_tab)

# read by region
def output_region_plot(test_predict,code,len_columns):
	tab_plot = []
	codes = {"11":0,"24":9,"27":18,"28":27,"32":36,"44":45,
	"52":54,"53":63,"5":72,"75":81,"76":90,"84":99,
	"93":108,"99":117}
	x = codes[code]
	p = x
	som = 0
	while x < len(test_predict[0]):
		som +=test_predict[0][x]
		if x == p+9:
			tab_plot.append(som)
			x +=len_columns-9
			p = x
			som = 0
		else:
			x+=1
	full_test_2D = []
	for v in tab_plot:
		full_test_2D.append([v])
	return full_test_2D

def output_full_plot(tab,code):
	tab_plot = []
	codes = {"11":0,"24":9,"27":18,"28":27,"32":36,"44":45,
	"52":54,"53":63,"5":72,"75":81,"76":90,"84":99,
	"93":108,"99":117}
	p = codes[code]
	print(p)
	som = 0
	for value in tab:
		som = 0
		for v in value[p:p+10]:
			print(v)
			som += float(v)
		print("out")
		tab_plot.append(som)
	full_mnt_2D = []
	for v in tab_plot:
		full_mnt_2D.append([v])
	return full_mnt_2D
#def return_one_value_for_one_time():

