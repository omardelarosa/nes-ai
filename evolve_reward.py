
def does_mutation_occur(self):
    #Probability of mutation
    mutation_prob = 0.1
    if random.randint(1,100)/100 <= mutation_prob:
        return True
    else:
        return False

def does_crossover_occur(self):
	
    crossover_prob = 0.7
    
    if random.randint(1,100)/100 <= crossover_prob:
        return True
    else:
        return False

def crossover(self,chromx,chromy,chrom_length):

    offspring = [None] * chrom_length
    prob_gene_from_chromx = 0.5
    for i in range(0,chrom_length):
        if random.randint(1,100)/100 <= prob_gene_from_chromx:
            offspring[i] = chromx[i]
        else:
            offspring[i] = chromy[i]

    return offspring

def mutate(self, chromosome, chrom_length):
    component_ind = random.randint(0,chrom_length-1)
    chromosome[component_ind] = (chromosome[component_ind] + 1) % 2

def fitness(chromosome):
	#Test reward def for model using chromosome here with reinforcement model
    return

chrom_length = 4
pop_size = 8
best_evaluation = -999999
best_reward_calc = [None]*chrom_length
population = [None] * pop_size

#initialize population
for i in range(0,pop_size):
    temp_seq = [None] * chrom_length
    for j in range(0,chrom_length):
        temp_seq[j] = possible[random.randint(0,len(possible)-1)]
    population[i] = temp_seq

t = 0
max_iter = 100
while(t > max_iter):
	rank = [None]*pop_size
	for i in range(0,pop_size):
		performance = fitness(population[i])
		rank[i] = (performance,i)
		if performance > best_evaluation:
			best_evaluation = performance
			best_reward_calc = population[i]

	#create distribution to select from
	rank.sort(key = lambda x:x[0], reverse = True)
	distribution = []
	for i in range(0,pop_size):
		for j in range(0,(pop_size - i)):
			distribution.append(rank[i][1])

	#Select members of population 
	selected = []
	for i in range(0,pop_size):
		selected_ind = distribution[random.randint(0,len(distribution)-1)]
		selected.append(population[selected_ind])

	next_gen = [None]*pop_size
	for i in range(0,pop_size,2):
		if self.does_crossover_occur() == True:
			next_gen[i] = self.crossover(selected[i],selected[i+1],chrom_length)
			next_gen[i+1] = self.crossover(selected[i],selected[i+1],chrom_length)
		else:
			next_gen[i] = selected[i]
			next_gen[i+1] = selected[i+1]

	for i in range(0,pop_size):
		if self.does_mutation_occur() == True:
			self.mutate(next_gen[i],chrom_length, possible)

	population = next_gen
	t+=1

print("Best Reward Calculation Method: ",best_reward_calc)
if best_reward_calc[0] == 1:
	print("Includes Score: Yes")
else:
	print("Includes Score: No")
if best_reward_calc[1] == 1:
	print("Includes Number of Lives: Yes")
else:
	print("Includes Number of Lives: No")
if best_reward_calc[2] == 1:
	print("Includes Mega Man Life Bar: Yes")
else:
	print("Includes Mega Man Life Bar: No")
if best_reward_calc[2] == 1:
	print("Includes Bonus Ball: Yes")
else:
	print("Includes Bonus Ball: No")




	

