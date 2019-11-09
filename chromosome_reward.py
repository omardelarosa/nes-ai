from heuristics import *

def calcChromReward(chromosome,ram_vector):
	#chromosome is vector of 1's and 0's
	return (chromosome[0]*calcScore(ram_vector) 
			+ chromosome[1]*calcNumLives(ram_vector) 
			+ chromsome[2]*calcMegamanLife(ram_vector)
			+ chromosome[3]*calcBonusBall(ram_vector))
