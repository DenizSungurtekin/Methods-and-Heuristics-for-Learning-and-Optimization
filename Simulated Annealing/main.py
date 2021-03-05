import numpy as np
import random
import copy
## Dont hesitate to comment code to execute the file partly

def distance(city, city2):  # Compute distance between two city
    return np.sqrt((city.posX - city2.posX) ** 2 + (city.posY - city2.posY) ** 2)


def pathLenght(value):  # Compute lenght of the path of a configuration which is a list of city (Energy)
    path_length = 0
    for i in range(len(value) - 1):
        path_length += distance(value[i], value[i + 1])
    return path_length


class City:
    def __init__(self, id, posX, posY):  # Data contains triplet [id,posX,posY] and index give the row number of a city
        self.id = id
        self.posX = posX
        self.posY = posY

    def who(self):
        print("I am the city with id: %s my position is %f,%f" % (self.id, self.posX, self.posY))

    def toListe(self):
        return [self.id, self.posX, self.posY]

    def toListe2(self):
        return [self.id]


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
        return [[element.id] for element in self.value]

    def increment(self):
        self.counter += 1

    def incrementAttempt(self):
        self.attemptedCounter += 1

    def incrementGeneralCounter(self):
        self.generalCounter += 1

    def reinitializeGeneralAndAttempt(self):
        self.attemptedCounter = 0
        self.generalCounter = 0

    def reinitializeCounter(self):
        self.counter = 0


def differenceEnergy(current, neighbor):  # Compute Variation
    return neighbor.energy - current.energy


def probabilitie(current, neighbor, T):  # Metropolis Rule
    return min([1, np.exp(-differenceEnergy(current, neighbor) / T)])


def acceptCondition(current, neighbor, T):  # Return true if we accept the new solution
    randomValue = random.uniform(0, 1)
    return randomValue < np.exp(-differenceEnergy(current, neighbor) / T)


def updateTemperature(T):  # Update the temperature
    return T * 0.9


def isFreezed(configuration):
    return configuration.counter == 3


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
    return [City(data[i][0], data[i][1], data[i][2]) for i in range(len(data))]


def transposing(configuration):  # Make the switch of 2 cities by generating two integer in the range depending of the number of cities
    voisin = copy.deepcopy(configuration)
    lenght = len(configuration.value)
    firstCityIndex = random.randint(0,
                                    lenght - 2)  # -2 because we dont want it to be the last city which is the first (cities[length-1])
    secondCityIndex = random.randint(0, lenght - 2)
    while firstCityIndex == secondCityIndex:  # In case we have same index
        secondCityIndex = random.randint(0, lenght - 2)

    voisin.value[firstCityIndex], voisin.value[secondCityIndex] = voisin.value[secondCityIndex], voisin.value[
        firstCityIndex]
    del voisin.value[-1]  # Because the first can be changed
    newValue = voisin.value
    newValue.append(newValue[0])

    newConfiguration = Configuration(newValue, configuration.generalCounter + 1, configuration.attemptedCounter + 1,
                                     configuration.Tvalues)
    return newConfiguration


def isEquilibrium(configuration, n):  # Equilibrium conditions
    if (configuration.generalCounter >= 12 * n) or (configuration.attemptedCounter >= 100 * n):
        return True


def isFreezing(configuration):
    if configuration.counter >= 4:
        return True


def solver(datname):
    # Step 1 Initial Configuration
    data = datToArray(datname)
    cities = affectCity(data)
    cities = random.sample(cities, len(cities))  # We shuffle to have a random cities configuration
    cities.append(cities[0])
    Tvalues = []
    initialConfiguration = Configuration(cities, 0, 0, Tvalues)

    # Step 2 Initial temperature
    energies = []
    configuration = copy.deepcopy(initialConfiguration)
    for i in range(100):
        neighbor = transposing(configuration)
        energy = differenceEnergy(configuration, neighbor)
        energies.append(energy)
        configuration = neighbor

    mean = np.mean(energies)  # Compute the mean value

    firstIteration = True
    temperatureBestFitness = 100000  # Initialize a big "random" fitness value

    while not isFreezing(initialConfiguration):  # Freezing conditions step 7

        if firstIteration:  # If its the first attempt we define T0  / With equation 1) we have T0 = -delta_E/ln(0.5) -> Possibly a negatif value so we use abs -> problem because we want T to be minimum at 0
            T = abs(-mean / np.log(0.5))
            Tvalues.append(T)
            firstIteration = False
        else:
            T = updateTemperature(T)  # Otherwise we update it -> Temperature reduction (Step 6)
            Tvalues.append(T)
        while not isEquilibrium(initialConfiguration, len(
                cities) - 1):  # Need to reinitialize attemptedCounter and general counter -> Step5 Equilibrium condition
            neighbor = transposing(
                initialConfiguration)  # Compute a random neighbor -> Elementary configuration update (Step 3)
            r = random.uniform(0,
                               1)  # Compute a random number between 0 and 1 to chose randomly if we take or not our neighbor

            if probabilitie(initialConfiguration, neighbor, T) >= r:  # Acceptance/Rejection rule
                initialConfiguration = neighbor  ## Counter of neighbor incremented during transposing
            else:
                initialConfiguration.incrementAttempt()  # Otherwise we increment only the attempt

        initialConfiguration.reinitializeGeneralAndAttempt()  # We reinitialize the equilibrium condition

        if temperatureBestFitness > initialConfiguration.energy:  # Check if there is improvement on the fitness during the temperature step
            initialConfiguration.reinitializeCounter()  # If yes we reinitialize the counter
            temperatureBestFitness = initialConfiguration.energy  # And we store it
        else:
            initialConfiguration.increment()  # Otherwise we increment it to know that at this temperature there isn't improvement

    return initialConfiguration


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
    return solution


# TEST
# Simulated Annealing
solution = solver("cities")  # Can take cities or cities2 in input
solution.finalResponse()

# Greedy
solution = GreedySolver("cities")  # Can take cities or cities2 in input
solution.finalResponse()


# Run 10 times each configuration
citiesSolutions = [solver("cities") for k in range(10)]
citiesSolutionsGreedy = [GreedySolver("cities") for k in range(10)]
citiesSolutions2 = [solver("cities2") for k in range(10)]
citiesSolutionsGreedy2 = [GreedySolver("cities2") for k in range(10)]

import matplotlib.pyplot as plt

# 10 result for cities.data file with SA
for element in citiesSolutions:
    print("My Configuration is: ")
    print(element.toListe())
    print("My path length is: ", element.energy)
    print(" ")

# 10 result for cities.data file with Greedy
for element in citiesSolutionsGreedy:
    print("My Configuration is: ")
    print(element.toListe())
    print("My path length is: ", element.energy)
    print(" ")

# 10 result for cities2.data file with SA
for element in citiesSolutions2:
    print("My Configuration is: ")
    print(element.toListe2())
    print("My path length is: ", element.energy)
    print(" ")

# 10 result for cities2.data file with SA
for element in citiesSolutionsGreedy2:
    print("My Configuration is: ")
    print(element.toListe2())
    print("My path length is: ", element.energy)
    print(" ")



# Generation list of randoms cities of size n
def generateList(n):
    cities = []
    for i in range(n):
        cities.append(City(i, random.uniform(0, 5), random.uniform(0, 5)))
    return cities


# 50 60 80 100
cities50 = generateList(50)
cities60 = generateList(60)
cities80 = generateList(80)
cities100 = generateList(100)


# modification on function solver to take directly a list of cities and return execution time
import time
def solver2(cities):
    # Step 1 Initial Configuration
    start = time.time()
    cities = random.sample(cities, len(cities))  # We shuffle to have a random cities configuration
    cities.append(cities[0])
    Tvalues = []
    initialConfiguration = Configuration(cities, 0, 0, Tvalues)

    # Step 2 Initial temperature
    energies = []
    configuration = copy.deepcopy(initialConfiguration)
    for i in range(100):
        neighbor = transposing(configuration)
        energy = differenceEnergy(configuration, neighbor)
        energies.append(energy)
        configuration = neighbor

    mean = np.mean(energies)  # Compute the mean value

    firstIteration = True
    temperatureBestFitness = 100000  # Initialize a big "random" fitness value

    while not isFreezing(initialConfiguration):  # Freezing conditions step 7

        if firstIteration:  # If its the first attempt we define T0  / With equation 1) we have T0 = -delta_E/ln(0.5) -> Possibly a negatif value so we use abs -> problem because we want T to be minimum at 0
            T = abs(-mean / np.log(0.5))
            Tvalues.append(T)
            firstIteration = False
        else:
            T = updateTemperature(T)  # Otherwise we update it -> Temperature reduction (Step 6)
            Tvalues.append(T)
        while not isEquilibrium(initialConfiguration, len(
                cities) - 1):  # Need to reinitialize attemptedCounter and general counter -> Step5 Equilibrium condition
            neighbor = transposing(
                initialConfiguration)  # Compute a random neighbor -> Elementary configuration update (Step 3)
            r = random.uniform(0,
                               1)  # Compute a random number between 0 and 1 to chose randomly if we take or not our neighbor

            if probabilitie(initialConfiguration, neighbor, T) >= r:  # Acceptance/Rejection rule
                initialConfiguration = neighbor  ## Counter of neighbor incremented during transposing
            else:
                initialConfiguration.incrementAttempt()  # Otherwise we increment only the attempt

        initialConfiguration.reinitializeGeneralAndAttempt()  # We reinitialize the equilibrium condition

        if temperatureBestFitness > initialConfiguration.energy:  # Check if there is improvement on the fitness during the temperature step
            initialConfiguration.reinitializeCounter()  # If yes we reinitialize the counter
            temperatureBestFitness = initialConfiguration.energy  # And we store it
        else:
            initialConfiguration.increment()  # Otherwise we increment it to know that at this temperature there isn't improvement
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
    return end - start


# Contains execution time for SA
times50Sa = [solver2(cities50) for k in range(10)]
times60Sa = [solver2(cities60) for k in range(10)]
times80Sa = [solver2(cities80) for k in range(10)]
times100Sa = [solver2(cities100) for k in range(10)]

# Contains execution time for Greedy
times50G = [GreedySolver2(cities50) for k in range(10)]
times60G = [GreedySolver2(cities60) for k in range(10)]
times80G = [GreedySolver2(cities80) for k in range(10)]
times100G = [GreedySolver2(cities100) for k in range(10)]

mean50Sa = np.mean(times50Sa)
mean60Sa = np.mean(times60Sa)
mean80Sa = np.mean(times80Sa)
mean100Sa = np.mean(times100Sa)

std50Sa = np.std(times50Sa)
std60Sa = np.std(times60Sa)
std80Sa = np.std(times80Sa)
std100Sa = np.std(times100Sa)

mean50G = np.mean(times50G)
mean60G = np.mean(times60G)
mean80G = np.mean(times80G)
mean100G = np.mean(times100G)

std50G = np.std(times50G)
std60G = np.std(times60G)
std80G = np.std(times80G)
std100G = np.std(times100G)

# plot of time depending of the size for SA
import matplotlib.pyplot as plt
sizes = [50, 60, 80, 100]
meansSa = [mean50Sa, mean60Sa, mean80Sa, mean100Sa]
meansG = [mean50G, mean60G, mean80G, mean100G]

plt.plot(sizes, meansSa, label="SA")
plt.plot(sizes, meansG, label="Greedy")
plt.xlabel("Size of the problem")
plt.ylabel("mean value of times")

plt.title("Executions times")
plt.legend()
plt.show()

#Print of mean and std value of times
print("mean times for SA")
print("size 50: ",mean50Sa)
print("size 60: ",mean60Sa)
print("size 80: ",mean80Sa)
print("size 100: ",mean100Sa)

print("mean times for Greedy")
print("size 50: ",mean50G)
print("size 60: ",mean60G)
print("size 80: ",mean80G)
print("size 100: ",mean100G)

print("std for SA")
print("size 50: ",std50Sa)
print("size 60: ",std60Sa)
print("size 80: ",std80Sa)
print("size 100: ",std100Sa)

print("std for Greedy")
print("size 50: ",std50G)
print("size 60: ",std60G)
print("size 80: ",std80G)
print("size 100: ",std100G)