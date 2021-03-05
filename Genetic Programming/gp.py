import random
import numpy as np
import matplotlib.pyplot as plt
# This is the machine on which programs are executed
# The output is the value on top of the pile.

class CPU:
    def __init__(self):
        self.pile=[]
    def reset(self):
        while len(self.pile)>0:self.pile.pop()

# These are the instructions
# For each operator we check if there is enought variable to use them, else we reset the pile. (If a pile is empty after an execution of an operator the program is considered as invalid)
def AND(cpu, data):
    if len(cpu.pile)<2:
        cpu.reset()
    else:
        arg1 = cpu.pile.pop()
        arg2 = cpu.pile.pop()
        cpu.pile.append(arg1 and arg2)


def OR(cpu, data):
    if len(cpu.pile)<2:
        cpu.reset()
    else:
        arg1 = cpu.pile.pop()
        arg2 = cpu.pile.pop()
        cpu.pile.append(arg1 or arg2)

def XOR(cpu, data):
    if len(cpu.pile)<2:
        cpu.reset()
    else:
        arg1 = cpu.pile.pop()
        arg2 = cpu.pile.pop()
        cpu.pile.append(arg1 ^ arg2)


def NOT(cpu, data):
    if len(cpu.pile)<1:
        cpu.reset()
    else:
        arg1 = cpu.pile.pop()
        if arg1 == 1:
            cpu.pile.append(0)
        else:
            cpu.pile.append(1)


# Push values of variables on the stack.
# Here we juste look the data and push the corresponding valu in the stack
def X1(cpu, data):
    cpu.pile.append(data[0])

def X2(cpu, data):
    cpu.pile.append(data[1])

def X3(cpu, data):
    cpu.pile.append(data[2])

def X4(cpu, data):
    cpu.pile.append(data[3])


# Here we juste do a switch case whith each instruction, if the instruction is empty we return a 2 which is the error number else we return the last element of the pile of size 1
def execute(program,cpu, data):
    if program.count("X1") > 1 or program.count("X2") > 1 or program.count("X3") > 1 or program.count("X4") > 1: ## Condition to prevent same variable declaration
        return 2
    for element in program:
        if element == "X1":
            X1(cpu,data)
            continue
        if element == "X2":
            X2(cpu,data)
            continue
        if element == "X3":
            X3(cpu,data)
            continue
        if element == "X4":
            X4(cpu,data)
            continue
        if element == "XOR":
            XOR(cpu,data)
            if cpu.pile == []:
                break
            else:
                continue

        if element == "NOT":
            NOT(cpu,data)
            if cpu.pile == []:
                break
            else:
                continue
        if element == "OR":
            OR(cpu,data)
            if cpu.pile == []:
                break
            else:
                continue
        else:
            AND(cpu,data)
            if cpu.pile == []:
                break
            else:
                continue

    if cpu.pile == [] or len(cpu.pile)>1: ## If the pile is empty, the program is illegal / if its len is bigger than 1 there is some useless declaration
        return 2
    else:
        return cpu.pile[-1]

# Generate a random program
# We juste take a random element in the concatenation of the two set
def randomProg(length,functionSet,terminalSet):
    progElement = functionSet + terminalSet
    prog = []
    for i in range(length):
        r = random.randint(0,len(progElement)-1)
        prog.append(progElement[r])
    return prog

# Computes the fitness of a program. 
# The fitness counts how many instances of data in dataSet are correctly computed by the program
# We just check how many time the last element of a data is equal to the execution of a program
def computeFitness(prog,cpu,dataSet):
    count = 0
    for data in dataSet:
        cpu.reset()
        out = execute(prog,cpu,data)
        if out == data[-1]:
            count += 1
    return count

    
# Selection using 2-tournament.
def selection(Population,cpu,dataSet):
    listOfFitness=[]
    for i in range(len(Population)):
        prog=Population[i]
        f=computeFitness(prog,cpu,dataSet)
        listOfFitness.append( (i,f) )

    newPopulation=[]
    n=len(Population)
    for i in range(n):    
        i1=random.randint(0,n-1)
        i2=random.randint(0,n-1)
        if listOfFitness[i1][1]>listOfFitness[i2][1]:
            newPopulation.append(Population[i1])
        else:
            newPopulation.append(Population[i2])
    return newPopulation

def crossover(Population,p_c):
    newPopulation=[]
    n=len(Population)
    i=0
    while(i<n):
        p1=Population[i]
        p2=Population[(i+1)%n]
        m=len(p1)
        if random.random()<p_c:  # crossover
            k=random.randint(1,m-1)
            newP1=p1[0:k]+p2[k:m]
            newP2=p2[0:k]+p1[k:m]
            p1=newP1
            p2=newP2
        newPopulation.append(p1)
        newPopulation.append(p2)
        i+=2
    return newPopulation

def mutation(Population,p_m,terminalSet,functionSet):
    newPopulation=[]
    nT=len(terminalSet)-1
    nF=len(functionSet)-1
    for p in Population:
        for i in range(len(p)):
            if random.random()>p_m:continue
            if random.random()<0.5: 
                p[i]=terminalSet[random.randint(0,nT)]
            else:
                p[i]=functionSet[random.randint(0,nF)]
        newPopulation.append(p)
    return newPopulation

#-------------------------------------


#Parameters
# Function and terminal sets.
dataSet=[[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,1,0],[0,1,0,0,0],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1],[1,0,0,0,0],[1,0,0,1,1],[1,0,1,0,0],[1,0,1,1,0],[1,1,0,0,0],[1,1,0,1,0],[1,1,1,0,0],[1,1,1,1,0]]
functionSet=["AND", "OR", "NOT", "XOR"]
terminalSet=["X1","X2","X3","X4"]
cpu=CPU()
popSize = 100
p_c = 0.6
p_m = 0.01
nbGen = 40
# Generate the initial population 

# Evolution. Loop on the creation of population at generation i+1 from population at generation i, through selection, crossover and mutation.

def solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet):
    bestfit = []
    population = [randomProg(5, functionSet, terminalSet) for i in range(popSize)]

    for i in range(nbGen):
        population = selection(population,cpu,dataSet)
        population = crossover(population,p_c)
        population = mutation(population,p_m,terminalSet,functionSet)
        fiteness = [computeFitness(individu,cpu,dataSet) for individu in population]

        maxIndex = np.argmax(fiteness)
        bestIndividu = population[maxIndex]
        bestfit.append(computeFitness(bestIndividu,cpu,dataSet))

    return bestIndividu,computeFitness(bestIndividu,cpu,dataSet),bestfit # Return the best individu, his fitness, and allbest fitness of each generation

##Uncomment to see an example of solution
print(solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet))

#Uncomment to see all best configuration with max fitness
#
# personnes = [solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet)[0] for i in range(3000) ]
# personnes = np.unique(personnes, axis=0)
# personnes = personnes.tolist()
# bestPersonnes = [element for element in personnes if computeFitness(element,cpu,dataSet) == 13]
# bestPersonnes = np.asarray(bestPersonnes)
# print(bestPersonnes)
# print("There is ",len(bestPersonnes)," configuration with fitness value of 13")

x = [i for i in range(nbGen)]

# Plot of average. std, bestfitnee on 10 run with three differents set of parameter

p_c = 0.6
p_m = 0.01

results = [solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet)[2] for i in range(10)]
mean = np.mean(results,axis = 0)
std = np.std(results,axis = 0)
best = np.amax(results,axis = 0)

p_c = 0.2
p_m = 0.01
results = [solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet)[2] for i in range(10)]
mean2 = np.mean(results,axis = 0)
std2 = np.std(results,axis = 0)
best2 = np.amax(results,axis = 0)

p_c = 0.6
p_m = 0.1
results = [solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet)[2] for i in range(10)]
mean3 = np.mean(results,axis = 0)
std3 = np.std(results,axis = 0)
best3 = np.amax(results,axis = 0)

p_c = 0.2
p_m = 0.1
results = [solve(cpu,popSize,p_c,p_m,nbGen,dataSet,functionSet,terminalSet)[2] for i in range(10)]
mean4 = np.mean(results,axis = 0)
std4 = np.std(results,axis = 0)
best4 = np.amax(results,axis = 0)

fig = plt.figure(1)
plt.plot(x,mean ,label = "p_c = 0.6, p_m = 0.01")
plt.plot(x,mean2, label = "p_c = 0.2, p_m = 0.01")
plt.plot(x,mean3, label = "p_c = 0.6, p_m = 0.1")
plt.plot(x,mean4, label = "p_c = 0.2, p_m = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("Mean value of energy over 10 run")
plt.legend()
plt.show()

fig = plt.figure(2)
plt.plot(x,std,label = "p_c = 0.6, p_m = 0.01")
plt.plot(x,std2, label = "p_c = 0.2, p_m = 0.01")
plt.plot(x,std3, label = "p_c = 0.6, p_m = 0.1")
plt.plot(x,std4, label = "p_c = 0.2, p_m = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("std value of energy over 10 run")
plt.legend()
plt.show()

fig = plt.figure(3)
plt.plot(x,best,label = "p_c = 0.6, p_m = 0.01")
plt.plot(x,best2, label = "p_c = 0.2, p_m = 0.01")
plt.plot(x,best3, label = "p_c = 0.6, p_m = 0.1")
plt.plot(x,best4, label = "p_c = 0.2, p_m = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("best value of energy over 10 run")
plt.legend()
plt.show()



