import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib import cm

def energy(x,y):
    return -abs(1/2*x*np.sin(np.sqrt(abs(x))))-abs(y*np.sin(30*np.sqrt(abs(x/y))))

def maping(x,m,a,b):
    return round((int(x,2)/(2**m))*(b-a)+a)


def crossOver(individu1,individu2): # One point crossover with midbreak policy
    r = rd.uniform(0,1)
    if r <= 0.6: #p_c value
        x1 = individu1.x
        x2 = individu2.x
        y1 = individu1.y
        y2 = individu2.y
        newSequence1 = x1+y2
        newSequence2 = x2+y1
        return Individu(newSequence1),Individu(newSequence2)
    else:
        return individu1,individu2

def mutation(individu,p): #take sequence of bits and p_m
    x = individu.sequence
    x2 = ""
    for i in range(len(x)):
        r = rd.uniform(0,1)
        if r <= p:
            if x[i] == "1":
                x2 += "0"
            else:
                x2 += "1"
        else:
            x2 += x[i]
    individu.sequence = x2
    return individu

class Individu():
    def __init__(self,x):
        self.sequence = x
        self.x = x[0:10]
        self.y = x[10:]
        self.energy = energy(maping(self.x,10,10,1000),maping(self.y,10,10,1000))

def generateIndividu(N): #Generate N individu with random sequence value and put them in a list
    Individus = []
    for i in range(N):
        x = ""
        y = ""
        for j in range(20): #Generate a random sequence of 20 bits containing x and y
            rx = rd.uniform(0,1)

            if rx <= 0.5:
                x += "1"
            else:
                x += "0"

        Individus.append(Individu(x))

    return Individus

def tournament(Individus): #Choose randomly 5 individu and return the best
    N = len(Individus)
    r = [rd.randint(0,N-1) for i in range(5)]
    while len(np.unique(r))<5: #In case there is two same random value
        r = [rd.randint(0, N-1) for i in range(5)]
    energies = [Individus[r[i]].energy for i in range(5)]
    bestIndex = np.argmin(energies)
    return Individus[r[bestIndex]]



# Plot of the function and print minimal value
x = [i for i in range(10,1001)]
y = x
X,Y = np.meshgrid(x,y)
Z = energy(X,Y)

fig = plt.figure(1)
ax = plt.axes(projection="3d")
ax.contour3D(X,Y,Z,50,cmap = cm.coolwarm, antialiased = False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Energy function")
plt.show()
print("Optimum value: ",np.min(Z)) #-1356.482683119401

# Genetic algorithm with crossover
def solve(N,p_m,ite):
    individus = generateIndividu(N)
    for i in range(ite):
        individus = [tournament(individus) for i in range(N)] #Selection
        for i in range(N): # Crossover
            if i % 2 != 0:
                individus[i-1],individus[i] = crossOver(individus[i-1],individus[i])
        for i in range(N): # mutation and update values because only sequences were changed
            individus[i] = mutation(individus[i],p_m)
            individus[i].x = individus[i].sequence[0:10]
            individus[i].y = individus[i].sequence[10:]
            individus[i].energy = energy(maping(individus[i].x, 10, 10, 1000),maping(individus[i].y, 10, 10, 1000))
    energies = [individus[i].energy for i in range(N)]
    return min(energies)

# Genetic algorithm without crossover
def solveNoCross(N,p_m,ite): #Same function without cross operation
    individus = generateIndividu(N)
    for i in range(ite):
        individus = [tournament(individus) for i in range(N)] #Selection

        for i in range(N): # mutation and update values because only sequences were changed
            individus[i] = mutation(individus[i],p_m)
            individus[i].x = individus[i].sequence[0:10]
            individus[i].y = individus[i].sequence[10:]
            individus[i].energy = energy(maping(individus[i].x, 10, 10, 1000),maping(individus[i].y, 10, 10, 1000))
    energies = [individus[i].energy for i in range(N)]
    return min(energies)



#1 The cumulative empirical probability
# Genetic algorithm with crossover to display energy at each step
def solveEmpiricalCross(N,p_m,ite):
    individus = generateIndividu(N)
    energies = []
    for i in range(ite):
        individus = [tournament(individus) for i in range(N)] #Selection
        for i in range(N): # Crossover
            if i % 2 != 0:
                individus[i-1],individus[i] = crossOver(individus[i-1],individus[i])
        for i in range(N): # mutation and update values because only sequences were changed
            individus[i] = mutation(individus[i],p_m)
            individus[i].x = individus[i].sequence[0:10]
            individus[i].y = individus[i].sequence[10:]
            individus[i].energy = energy(maping(individus[i].x, 10, 10, 1000),maping(individus[i].y, 10, 10, 1000))
        energie = [individus[i].energy for i in range(N)]
        energies.append(energie)
    return energies

def solveEmpiricalNoCross(N,p_m,ite):
    individus = generateIndividu(N)
    energies = []
    for i in range(ite):
        individus = [tournament(individus) for i in range(N)] #Selection

        for i in range(N): # mutation and update values because only sequences were changed
            individus[i] = mutation(individus[i],p_m)
            individus[i].x = individus[i].sequence[0:10]
            individus[i].y = individus[i].sequence[10:]
            individus[i].energy = energy(maping(individus[i].x, 10, 10, 1000),maping(individus[i].y, 10, 10, 1000))

        energie = [individus[i].energy for i in range(N)]
        energies.append(energie)
    return energies

minimum = -1356.482683119401

#Function that count the number of optimum value in a energies list and divide by 100
def countOpti(energies):
    return sum(map(lambda x : x==minimum,energies))/len(energies)

#Same for 1%
def count1percent(energies):
    err = minimum/100
    value = -1356.482683119401-err
    return sum(map(lambda x : x<value,energies))/len(energies)

#Same for 2.5%
def count2percent(energies):
    err = (minimum/100)*2.5
    value = -1356.482683119401-err
    return sum(map(lambda x : x<value,energies))/len(energies)


#Plot of cumulative empirical probability
x = [i for i in range(100)]

result = [solveEmpiricalCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [countOpti(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.1")

result = [solveEmpiricalCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [countOpti(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.01")

result = [solveEmpiricalNoCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [countOpti(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.1")

result = [solveEmpiricalNoCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [countOpti(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.01")

plt.xlabel("Iterations")
plt.ylabel("Cumulative empirical probability with optimum value")
plt.legend()
plt.show()



result = [solveEmpiricalCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count1percent(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.1")

result = [solveEmpiricalCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count1percent(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.01")

result = [solveEmpiricalNoCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count1percent(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.1")

result = [solveEmpiricalNoCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count1percent(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.01")

plt.xlabel("Iterations")
plt.ylabel("Cumulative empirical probability within 1% of the optimum value")
plt.legend()
plt.show()



result = [solveEmpiricalCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count2percent(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.1")

result = [solveEmpiricalCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count2percent(element) for element in mean]
plt.plot(x,cumul,label="Cross, p_m = 0.01")

result = [solveEmpiricalNoCross(100,0.1,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count2percent(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.1")

result = [solveEmpiricalNoCross(100,0.01,100) for i in range(5)]
result = np.asarray(result)
mean = np.mean(result,axis=0) #Mean value because we had to run it several time to have different seed (random)
cumul = [count2percent(element) for element in mean]
plt.plot(x,cumul,label="No Cross, p_m = 0.01")

plt.xlabel("Iterations")
plt.ylabel("Cumulative empirical probability within 2,5% of the optimum value")
plt.legend()
plt.show()










#2. report best, average and std of fitness with ite = 10,100,1000 => because of fitnesse evaluation estimation with N = 100 and pm = [0.01,0.1]
iteration = [10,100,1000]
p_ms = [0.01,0.1]
resultatsCross = []
resultatsNoCross = []
for k in iteration:
    for p_m in p_ms:
        result = [solve(100,p_m,k) for i in range(10)]
        resultatsCross.append(result)
        result = [solveNoCross(100,p_m,k)]
        resultatsNoCross.append(result)

bestsCrossPm1 = [min(resultatsCross[i]) for i in range(3)]
bestsNoCrossPm1 = [min(resultatsNoCross[i]) for i in range(3)]
bestsCrossPm2 = [min(resultatsCross[i]) for i in range(3,6)]
bestsNoCrossPm2 = [min(resultatsNoCross[i]) for i in range(3,6)]

meanCrossPm1 = [np.mean(resultatsCross[i]) for i in range(3)]
meanNoCrossPm1 = [np.mean(resultatsNoCross[i]) for i in range(3)]
meanCrossPm2 = [np.mean(resultatsCross[i]) for i in range(3,6)]
meanNoCrossPm2 = [np.mean(resultatsNoCross[i]) for i in range(3,6)]

varCrossPm1 = [np.var(resultatsCross[i]) for i in range(3)]
varNoCrossPm1 = [np.var(resultatsNoCross[i]) for i in range(3)]
varCrossPm2 = [np.var(resultatsCross[i]) for i in range(3,6)]
varNoCrossPm2 = [np.var(resultatsNoCross[i]) for i in range(3,6)]

#Plot
x = iteration

fig = plt.figure(2)
plt.plot(x,bestsCrossPm1,marker = 'o',label = "Cross, pm = 0.01")
plt.plot(x,bestsNoCrossPm1,marker = 'o',label= "NoCross, pm = 0.01")
plt.plot(x,bestsCrossPm2,marker = 'o',label = "Cross, pm = 0.1")
plt.plot(x,bestsNoCrossPm2,marker = 'o',label= "NoCross, pm = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("Best energy among 10 run")
plt.legend()
plt.title("Best energies among 10 run in the four differents configurations")
plt.show()

fig = plt.figure(3)
plt.plot(x,meanCrossPm1,marker = 'o',label = "Cross, pm = 0.01")
plt.plot(x,meanNoCrossPm1,marker = 'o',label= "NoCross, pm = 0.01")
plt.plot(x,meanCrossPm2,marker = 'o',label = "Cross, pm = 0.1")
plt.plot(x,meanNoCrossPm2,marker = 'o',label= "NoCross, pm = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("Mean among 10 run")
plt.legend()
plt.title("Mean among 10 run in the four differents configurations")
plt.show()

fig = plt.figure(4)
plt.plot(x,varCrossPm1,marker = 'o',label = "Cross, pm = 0.01")
plt.plot(x,varNoCrossPm1,marker = 'o',label= "NoCross, pm = 0.01")
plt.plot(x,varCrossPm2,marker = 'o',label = "Cross, pm = 0.1")
plt.plot(x,varNoCrossPm2,marker = 'o',label= "NoCross, pm = 0.1")
plt.xlabel("Number of Generations")
plt.ylabel("Variance among 10 run")
plt.legend()
plt.title("Variance among 10 run in the four differents configurations")
plt.show()



