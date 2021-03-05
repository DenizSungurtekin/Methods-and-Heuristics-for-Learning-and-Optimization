import random
import numpy as np
import operator
import matplotlib.pyplot as plt
import time


class Graph(object):
    def __init__(self, cost_matrix: list, N: int):
        self.matrix = cost_matrix
        self.N = N
        self.pheromone = [[1 / (N * N) for j in range(N)] for i in range(N)]


class AntSytem(object):
    def __init__(self, nbr_fourmi: int, iterations: int, alpha: float, beta: float, rho: float, Q: int):

        self.Q = Q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.nbr_fourmi = nbr_fourmi
        self.iterations = iterations

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph: Graph):
        best_cost = float('inf')
        best_solution = []
        for ite in range(self.iterations):
            ants = [_Ant(self, graph) for i in range(self.nbr_fourmi)]
            for ant in ants:
                for i in range(graph.N - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu

                # update pheromone
                ant._update_pheromone_delta()

            self._update_pheromone(graph, ants)
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: AntSytem, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.N)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.N)] for i in range(graph.N)]  # heuristic information
        start = random.randint(0, graph.N - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                i] ** self.colony.beta
        probabilities = [0 for i in range(self.graph.N)]  # probabilities
        for i in range(self.graph.N):
            try:
                self.allowed.index(i)
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                                   self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass

        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.N)] for i in range(self.graph.N)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost

def plot(points, path: list): ## Used to plots all points and the corresponding path of a solution
    path.append(path[0]) # To make the plot circular
    x = [element[0] for element in points]
    y = [element[1] for element in points]

    y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    plt.plot(x, y, 'co')

    for _ in range(1, len(path)):
        i = path[_ - 1]
        j = path[_]
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='g', length_includes_head=True)

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Visualization of the path")

    plt.show()

def distance(city, city2):  # Compute distance between two city
    return np.sqrt((city.posX - city2.posX) ** 2 + (city.posY - city2.posY) ** 2)

class City:
    def __init__(self, id, posX, posY):  # Data contains triplet [id,posX,posY] and id give the row number of a city
        self.id = id
        self.posX = posX
        self.posY = posY

    def who(self):
        print("I am the city with id: %s my position is %f,%f" % (self.id, self.posX, self.posY))

    def toListe(self):
        return [self.id, self.posX, self.posY]

    def toListe2(self):
        return [self.id]

def is_float(string):
    try:
        return float(string)
    except ValueError:
        return False


# Function that read a dat file by inputing its name
def datToArray(fileName):
    # Read dat file and put element in strings format inside an array
    data = []
    name = fileName + ".dat"
    with open(name, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([float(i) if is_float(i) else i for i in k])

    data = np.array(data, dtype='O')

    # Read each row to convert into a tripler: [string,float,float]
    datas = []  # Will contain new format
    for i in range(len(data)):
        element = data[i][0].split()
        element[1] = float(element[1])
        element[2] = float(element[2])
        datas.append(element)

    return datas


def affectCity(data):  ## Store each city in a liste
    return [City(i, data[i][1], data[i][2]) for i in range(len(data))]

def pathLenght(value):  # Compute lenght of the path of a configuration which is a list of city (Energy)
    path_length = 0
    for i in range(len(value) - 1):
        path_length += distance(value[i], value[i + 1])
    return path_length

class Configuration:
    def __init__(self, value, generalCounter, attemptedCounter, Tvalues):
        self.value = value  ## list of cities : [c1,..,cn,c1]
        self.counter = 0  ## Number of time the configuration hasnt imporved during a temperature step (Freezing condition)
        self.energy = pathLenght(value)  ## Energy is the lenght of the path of a list of cities
        self.generalCounter = generalCounter  ## Number of time there is a change of value (Equilibrium first condition)
        self.attemptedCounter = attemptedCounter  ## Number of time of perturbation atempted
        self.Tvalues = Tvalues

    def who(self):
        print("This is the cities of this configuration: ")
        for element in self.value:
            element.who()
            print(" ")
        print("My path lenght is: ", self.energy, " and my counter is: ", self.counter)

    def finalResponse(self):
        print("The final solution is: ", [element.id for element in self.value], " with length: ", self.energy)

    def toListe(self):
        return [[element.id, element.posX, element.posY] for element in self.value]

    def toListe2(self):
        return [element.id for element in self.value]



def Antsolver(datname,m,tmax):
    data = datToArray(datname)
    cities = affectCity(data)
    cost_matrix = []
    N = len(cities)

    for i in range(N):
        row = []
        for j in range(N):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)
    L_nn = GreedySolver(datname)[1] # Computing t_0 and Q  paramaters determined by using Greed Algorithm
    aco = AntSytem(m, tmax, 1, 5, 1/L_nn, L_nn) #fourmies,ite,alpha,beta,t_0,Q
    graph = Graph(cost_matrix, N)
    path, cost = aco.solve(graph)
    return path,cost



def GreedySolver(datname):
    data = datToArray(datname)
    cities = affectCity(data)
    cities = random.sample(cities, len(cities))  # We shuffle
    city = cities[0]  # We take the first city after a shuffle
    result = []  # We store the result in this array
    result.append(city)  # Store the initial city
    del cities[0]  # Delete the corresponding city because we want to stop when this list is empty
    while cities != []:
        distances = []
        for element in cities:  # Compute distance beteween current city and all others
            distances.append(distance(city, element))
        indexes = np.argsort(distances)  # Sort by indexes
        result.append(cities[indexes[0]])  # Add the nearest neighbor
        city = cities[indexes[0]]  # Update our current city
        del cities[indexes[0]]  # Delete it from the list of possible cities

    result.append(result[0])  # We add the first element to the end
    solution = Configuration(result, 0, 0, [])
    return solution.toListe2(),solution.energy

def allPoints(datname): ## Function that take all points of the data
    data = datToArray(datname)
    cities = affectCity(data)
    points = [[element.posX,element.posY] for element in cities]
    return points




# Run AS algortihme on the two problems with plot:

res = Antsolver("cities",20,100)
print("Corresponding Path:Energy, ",res)
plot(allPoints("cities"),res[0])


res = Antsolver("cities2",20,100)
print("Corresponding Path:Energy, ",res)
plot(allPoints("cities2"),res[0])



# Observing importance of m and ite parameters:

m = [1,2,5,10,15,20]
res = [Antsolver("cities",element,100)[1] for element in m]
res2 = [Antsolver("cities2",element,100)[1] for element in m]
plt.plot(m,res,label="cities")
plt.plot(m,res2,label="cities2")
plt.xlabel("m values")
plt.ylabel("Energy")
plt.legend()
plt.title(" Influence of m with 100 iterations")
plt.show()
# After m = 10 the result seems not to change so we will observe iteration influence with m = 10

ite = [10,50,100,150,200]
res = [Antsolver("cities",10,element)[1] for element in ite]
res2 = [Antsolver("cities2",10,element)[1] for element in ite]
plt.plot(ite,res,label="cities")
plt.plot(ite,res2,label="cities2")
plt.xlabel("iterations values")
plt.ylabel("Energy")
plt.legend()
plt.title(" Influence of iteration with 10 ants")
plt.show()
# surprisingly the number of iteration doesn't really seem to increase the quality of our result best result with ite = 10 in our case

#Best parameters semms to be m=10,ite=10. Comparing ten result with m=10 and ite = 10 to Greed

# Run 10 times and comparing our algorithms on cities and cities2
#Cities

res = [Antsolver("cities",10,10) for i in range(10)]
resGreedy = [GreedySolver("cities") for i in range(10)]
x = [i for i in range(10)]
y = [element[1] for element in res]
yGreedy = [element[1] for element in resGreedy]
plt.plot(x,y,label="AS")
plt.plot(x,yGreedy,label="Greedy")
plt.ylabel("Energy")
plt.legend()
plt.title("Comparing AS to Greedy algorithm in cities.dat")
plt.show()
#printing Best solution
points = allPoints("cities")
indexes = np.argsort(y)
indexes2 = np.argsort(yGreedy)

plot(points,res[indexes[0]][0])
plot(points,resGreedy[indexes[0]][0])

#Cities2

res = [Antsolver("cities2",10,10) for i in range(10)]
resGreedy = [GreedySolver("cities2") for i in range(10)]
x = [i for i in range(10)]
y = [element[1] for element in res]
yGreedy = [element[1] for element in resGreedy]
plt.plot(x,y,label="AS")
plt.plot(x,yGreedy,label="Greedy")
plt.ylabel("Energy")
plt.legend()
plt.title("Comparing AS to Greedy algorithm in cities2.dat")
plt.show()
#printing Best solution
points = allPoints("cities2")
indexes = np.argsort(y)
indexes2 = np.argsort(yGreedy)

plot(points,res[indexes[0]][0])
plot(points,resGreedy[indexes[0]][0])

# Impact of problem size on execution time:
# And PLOT

#Here i redefine my solver function: they take now a list of cities and return the execution times -> easier for computing times
def Antsolver2(cities,m,tmax):
    start = time.time()
    cost_matrix = []
    N = len(cities)

    for i in range(N):
        row = []
        for j in range(N):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)
    L_nn = GreedySolver2(cities)[0] # Computing t_0 and Q  paramaters determined by using Greed Algorithm
    aco = AntSytem(m, tmax, 1, 5, 1/L_nn, L_nn) #fourmies,ite,alpha,beta,t_0,Q
    graph = Graph(cost_matrix, N)
    path, cost = aco.solve(graph)
    end = time.time()
    return end - start

def GreedySolver2(cities):
    start = time.time()
    cities = random.sample(cities, len(cities))  # We shuffle
    city = cities[0]  # We take the first city after a shuffle
    result = []  # We store the result in this array
    result.append(city)  # Store the initial city
    del cities[0]  # Delete the corresponding city because we want to stop when this list is empty
    while cities != []:
        distances = []
        for element in cities:  # Compute distance beteween current city and all others
            distances.append(distance(city, element))
        indexes = np.argsort(distances)  # Sort by indexes
        result.append(cities[indexes[0]])  # Add the nearest neighbor
        city = cities[indexes[0]]  # Update our current city
        del cities[indexes[0]]  # Delete it from the list of possible cities

    result.append(result[0])  # We add the first element to the end
    solution = Configuration(result, 0, 0, [])
    end = time.time()
    return solution.energy,end-start

# function that generate random cities of size n
def generateList(n):
    cities = []
    for i in range(n):
        cities.append(City(i, random.uniform(0, 5), random.uniform(0, 5)))
    return cities
#Compute different sample
problemeSize = [50,60,80,100]
timesIteAS = []
timesIteGreedy = []
for i in range(10):
    timesAS = [Antsolver2(generateList(element),10,10) for element in problemeSize]
    timesGreedy = [GreedySolver2(generateList(element))[1] for element in problemeSize]
    timesIteAS.append(timesAS)
    timesIteGreedy.append(timesGreedy)

meanAs50 = np.mean([element[0] for element in timesIteAS])
meanG50 = np.mean([element[0] for element in timesIteGreedy])
stdAs50 = np.std([element[0] for element in timesIteAS])
stdG50 = np.std([element[0] for element in timesIteGreedy])

meanAs60 = np.mean([element[1] for element in timesIteAS])
meanG60 = np.mean([element[1] for element in timesIteGreedy])
stdAs60 = np.std([element[1] for element in timesIteAS])
stdG60 = np.std([element[1] for element in timesIteGreedy])

meanAs80 = np.mean([element[2] for element in timesIteAS])
meanG80 = np.mean([element[2] for element in timesIteGreedy])
stdAs80 = np.std([element[2] for element in timesIteAS])
stdG80 = np.std([element[2] for element in timesIteGreedy])


meanAs100 = np.mean([element[3] for element in timesIteAS])
meanG100 = np.mean([element[3] for element in timesIteGreedy])
stdAs100 = np.std([element[3] for element in timesIteAS])
stdG100 = np.std([element[3] for element in timesIteGreedy])


meansAs = [meanAs50,meanAs60,meanAs80,meanAs100]
meansG = [meanG50,meanG60,meanG80,meanG100]

stdAs = [stdAs50,stdAs60,stdAs80,stdAs100]
stdG = [stdG50,stdG60,stdG80,stdG100]

#Plot
plt.plot(problemeSize,meansAs,label="Mean values for each size, AS")
plt.plot(problemeSize,meansG,label="Mean values for each size, Greedy")
plt.xlabel("Size")
plt.ylabel("Mean")
plt.legend()
plt.title("Mean of execution time over 10 execution")
plt.show()

plt.plot(problemeSize,stdAs,label="std values for each size, AS")
plt.plot(problemeSize,stdG,label="std values for each size, Greedy")
plt.xlabel("Size")
plt.ylabel("Std")
plt.legend()
plt.title("std of execution time over 10 execution")
plt.show()