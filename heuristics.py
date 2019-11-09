
def calcScore(ram_vector):
	return (ram_vector[114]* 1 + ram_vector[115]* 10 + ram_vector[116]* 100 
			+ ram_vector[117]*1000 + ram_vector[118]*10000 + ram_vector[119]*100000
			+ ram_vector[120]*1000000 + ram_vector[121]*10000000)

def calcNumLives(ram_vector):
	return  1000*ram_vector[166]


def calcMegamanLife(ram_vector):
	return 10*ram_vector[106]


def calcBonusBall(ram_vector):
	return ram_vector[174]
	

