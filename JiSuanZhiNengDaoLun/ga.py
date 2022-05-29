
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib
matplotlib.rc("font", family='SimHei')


DNA_LENGTH = 20
POPSIZE = 300
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
		mutation(child,mode=mutation_mode)
		new_pop.append(child)

	return new_pop

def mutation(child, MUTATION_PROBABILITY=0.01,mode='unit'):
	#按照个体变异
	if mode=='unit':
		if np.random.rand() < MUTATION_PROBABILITY:  
			place = np.random.randint(0, DNA_LENGTH*2) 
			child[place] = child[place] ^ 1 
	# 按照基因位变异
	elif mode=='genetic_locus':
		for i in range(len(child)):
			if np.random.rand() < MUTATION_PROBABILITY:
				child[i] = child[i] ^ 1
				

def select(pop, fitness,mode='roulette'):  
    #轮盘赌选择
    if mode == 'roulette':
        idx = np.random.choice(np.arange(POPSIZE), size=POPSIZE, replace=True,
                            p=(fitness)/(fitness.sum()))
        return pop[idx]
    #二进制锦标赛选择
    elif mode=='binary':
        id = []
        x = [random.randint(0, int(pop.shape[0])-1)for i in range(2*int(pop.shape[0]))]
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


if __name__ == "__main__":
	allbest=[]#记录进化出的最优解
	allbest_site=[]
	nowbest=[]#记录当前代中的最优解
	nowbest_site=[]
	pop = np.random.randint(2, size=(POPSIZE, DNA_LENGTH*2)) 
	site,fun=get_best(pop)
	allbest.append(fun)
	allbest_site.append(site)
	nowbest.append(fun)
	nowbest_site.append(site)
	for i in range(GENERATION): 
		x, y = transcode(pop)#解码
		pop = np.array(crossover_mutation(pop, CROSSOVER_PROBABILITY,mutation_mode='genetic_locus'))#交叉变异
		fitness = get_fit(pop)#计算适应度
		pop = select(pop, fitness,mode='binary')#选择
		#记录最优解
		site, fun = get_best(pop)
		nowbest.append(fun)
		nowbest_site.append(site)
		allbest.append(max(nowbest))
		allbest_site.append(nowbest_site[np.argmax(nowbest)])

	plt.figure(1)
	plt.xlabel('generation')
	plt.ylabel('f(x1,x2)')
	plt.plot(allbest, 'g', label='所有代中最好个体')
	plt.plot(nowbest, ':', label='当前代中最好个体')
	plt.text(60,37.4,'最大值：'+str(allbest[-1]), fontsize=15, color='g')
	plt.text(60,37.1,'最大值坐标：'+str(allbest_site[-1]), fontsize=15, color='g')
	plt.legend(loc=0)


	fig = plt.figure(2)
	ax = Axes3D(fig)
	X1 = np.linspace(*X1_RANGE, 100)
	X2 = np.linspace(*X2_RANGE, 100)
	X1, X2 = np.meshgrid(X1, X2)
	Y = FUNCTION(X1, X2)
	ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap='rainbow',alpha=0.1)
	ax.set_zlim(-10, 50)
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('f(x1,x2)')

	ax.plot(np.array(allbest_site)[:, 0], np.array(
		allbest_site)[:, 1], allbest, 'go-', label='最优个体')

	ax.legend()
	plt.show()
