# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib
matplotlib.rc("font", family='SimHei')


DNA_LENGTH = 20
POPSIZE = 200
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.003
GENERATION = 200
X1_RANGE = [-3, 12.1]
X2_RANGE = [4.1, 5.8]


def FUNCTION(x1, x2):
	# 目标函数为max f(x1,x2)=21.5+x1*sin(4pi*x1)+x2*sin(20pi*x2)
	#                 s.t.
	#                     -3.0<=x1<=12.1
	#                      4.1<=x2<=5.8
	return 21.5+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)


def get_fit(pop):
    x, y = transcode(pop)
    z = FUNCTION(x, y)
    return (z - np.min(z)) + 1e-3


def transcode(pop):
	a = pop.shape[1]/2
	x_pop = pop[:, :int(a)]
	y_pop = pop[:, int(a):]

	x = x_pop.dot(2**np.arange(DNA_LENGTH)
	              [::-1])/float(2**DNA_LENGTH-1)*(X1_RANGE[1]-X1_RANGE[0])+X1_RANGE[0]
	y = y_pop.dot(2**np.arange(DNA_LENGTH)
	              [::-1])/float(2**DNA_LENGTH-1)*(X2_RANGE[1]-X2_RANGE[0])+X2_RANGE[0]
	return x, y


def crossover_mutation(pop, CROSSOVER_PROBABILITY=0.8, mutation_mode='unit'):
	new_pop = []
	for father in pop:
		child = father
		if np.random.rand() < CROSSOVER_PROBABILITY:
			mother = pop[np.random.randint(POPSIZE)]
			cross_points = np.random.randint(low=0, high=DNA_LENGTH*2)
			child[cross_points:] = mother[cross_points:]
		mutation(child, mode=mutation_mode)
		new_pop.append(child)

	return new_pop


def mutation(child, MUTATION_PROBABILITY=0.01, mode='unit'):
	#按照个体变异
	if mode == 'unit':
		if np.random.rand() < MUTATION_PROBABILITY:
			place = np.random.randint(0, DNA_LENGTH*2)
			child[place] = child[place] ^ 1
	# 按照基因位变异
	elif mode == 'genetic_locus':
		for i in range(len(child)):
			if np.random.rand() < MUTATION_PROBABILITY:
				child[i] = child[i] ^ 1


def select(pop, fitness, mode='roulette'):
    #轮盘赌选择
    if mode == 'roulette':
        idx = np.random.choice(np.arange(POPSIZE), size=POPSIZE, replace=True,
                               p=(fitness)/(fitness.sum()))
        return pop[idx]
    #二进制锦标赛选择
    elif mode == 'binary':
        id = []
        x = [random.randint(0, int(pop.shape[0])-1)
             for i in range(2*int(pop.shape[0]))]
        for i in range(int(pop.shape[0])):
            j = x[2*i]
            k = x[2*i+1]
            if fitness[j] > fitness[k]:
                id.append(j)
            else:
                id.append(k)
        return pop[id]


def get_best(pop):
	fitness = get_fit(pop)
	max_fitness_index = np.argmax(fitness)
	x, y = transcode(pop)
	return (x[max_fitness_index], y[max_fitness_index]), FUNCTION(x[max_fitness_index], y[max_fitness_index])




# %%

bestjie_lun_ji=[]
for j in range(20):
	allbest = []  # 记录进化出的最优解
	allbest_site = []
	nowbest = []  # 记录当前代中的最优解
	nowbest_site = []
	pop = np.random.randint(2, size=(POPSIZE, DNA_LENGTH*2))
	site, fun = get_best(pop)
	allbest.append(fun)
	allbest_site.append(site)
	nowbest.append(fun)
	nowbest_site.append(site)
	for i in range(GENERATION):
		x, y = transcode(pop)
		pop = np.array(crossover_mutation(
			pop, CROSSOVER_PROBABILITY, mutation_mode='genetic_locus'))
		fitness = get_fit(pop)
		pop = select(pop, fitness, mode='roulette')
		site, fun = get_best(pop)
		nowbest.append(fun)
		nowbest_site.append(site)
		allbest.append(max(nowbest))
		allbest_site.append(nowbest_site[np.argmax(nowbest)])
	bestjie_lun_ji.append(allbest[-1])



# %%

bestjie_er_ji = []
for j in range(20):
	allbest = []  # 记录进化出的最优解
	allbest_site = []
	nowbest = []  # 记录当前代中的最优解
	nowbest_site = []
	pop = np.random.randint(2, size=(POPSIZE, DNA_LENGTH*2))
	site, fun = get_best(pop)
	allbest.append(fun)
	allbest_site.append(site)
	nowbest.append(fun)
	nowbest_site.append(site)
	for i in range(GENERATION):
		x, y = transcode(pop)
		pop = np.array(crossover_mutation(
			pop, CROSSOVER_PROBABILITY, mutation_mode='genetic_locus'))
		fitness = get_fit(pop)
		pop = select(pop, fitness, mode='binary')
		site, fun = get_best(pop)
		nowbest.append(fun)
		nowbest_site.append(site)
		allbest.append(max(nowbest))
		allbest_site.append(nowbest_site[np.argmax(nowbest)])
	bestjie_er_ji.append(allbest[-1])


# %%

bestjie_lun_ge = []
for j in range(20):
	allbest = []  # 记录进化出的最优解
	allbest_site = []
	nowbest = []  # 记录当前代中的最优解
	nowbest_site = []
	pop = np.random.randint(2, size=(POPSIZE, DNA_LENGTH*2))
	site, fun = get_best(pop)
	allbest.append(fun)
	allbest_site.append(site)
	nowbest.append(fun)
	nowbest_site.append(site)
	for i in range(GENERATION):
		x, y = transcode(pop)
		pop = np.array(crossover_mutation(
			pop, CROSSOVER_PROBABILITY, mutation_mode='unit'))
		fitness = get_fit(pop)
		pop = select(pop, fitness, mode='roulette')
		site, fun = get_best(pop)
		nowbest.append(fun)
		nowbest_site.append(site)
		allbest.append(max(nowbest))
		allbest_site.append(nowbest_site[np.argmax(nowbest)])
	bestjie_lun_ge.append(allbest[-1])


# %%

bestjie_er_ge = []
for j in range(20):
	allbest = []  # 记录进化出的最优解
	allbest_site = []
	nowbest = []  # 记录当前代中的最优解
	nowbest_site = []
	pop = np.random.randint(2, size=(POPSIZE, DNA_LENGTH*2))
	site, fun = get_best(pop)
	allbest.append(fun)
	allbest_site.append(site)
	nowbest.append(fun)
	nowbest_site.append(site)
	for i in range(GENERATION):
		x, y = transcode(pop)
		pop = np.array(crossover_mutation(
			pop, CROSSOVER_PROBABILITY, mutation_mode='unit'))
		fitness = get_fit(pop)
		pop = select(pop, fitness, mode='binary')
		site, fun = get_best(pop)
		nowbest.append(fun)
		nowbest_site.append(site)
		allbest.append(max(nowbest))
		allbest_site.append(nowbest_site[np.argmax(nowbest)])
	bestjie_er_ge.append(allbest[-1])
	


# %%
plt.figure(1)
plt.plot(bestjie_lun_ge, label='轮盘赌+个体变异')
plt.plot(bestjie_lun_ji,label='轮盘赌+基因位变异')
plt.plot(bestjie_er_ge,label='二进制+个体变异')
plt.plot(bestjie_er_ji,label='二进制+基因位变异')
plt.legend()
plt.xlabel('times')
plt.ylabel('f(x1,x2)')



# %%
plt.figure(2)
plt.plot(sum(bestjie_lun_ge)/len(bestjie_lun_ge) *
         np.ones(len(bestjie_lun_ge)), label='轮盘赌+个体变异')
plt.plot(sum(bestjie_lun_ji)/len(bestjie_lun_ge) *
         np.ones(len(bestjie_lun_ge)), label='轮盘赌+基因位变异')
plt.plot(sum(bestjie_er_ge)/len(bestjie_lun_ge) *
         np.ones(len(bestjie_lun_ge)), label='二进制+个体变异')
plt.plot(sum(bestjie_er_ji)/len(bestjie_lun_ge) *
         np.ones(len(bestjie_lun_ge)), label='二进制+基因位变异')
plt.legend()
plt.xlabel('times')
plt.ylabel('f(x1,x2)')
plt.show()


# %%
sum(bestjie_lun_ge)/len(bestjie_lun_ge)*np.ones(len(bestjie_lun_ge))



