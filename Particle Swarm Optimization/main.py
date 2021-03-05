import numpy as np
import random
import matplotlib.pyplot as plt

def readData(name):
    datas = []
    m = []
    n = []
    with open(name,'r') as file:
        d = file.readlines()
        for i in d:
            m.append(i)
        for element in m:
            n.append(list(element.split(",")))
        for line in n:
            ligne = []
            for element in line:
                ligne.append(float(element))
            datas.append(ligne)
    return datas


print("Reading data...")

X = readData("X.dat")
Y = readData("Y.dat")
Y = [item for sublist in Y for item in sublist]
X = np.asarray(X)
Y = np.asarray(Y) #1 if is a 2, 0 if is a 3

print("Shape of X: ",X.shape)
print("Shape of Y: ",Y.shape)

def sigmoid(z):
    try:
        return 1/(1+np.exp(-z))
    except TypeError:
        return np.asarray([sigmoid(element) for element in z])




# PSO

def matrixToVector(omega1,omega2):  #Function to transform our two matrix in a single vector, s
    flat1 = omega1.reshape(25*(400+1))
    flat2 = omega2.reshape(25+1)
    flat1 = np.append(flat1,flat2)
    return flat1

def VectorToMatrix(s): #Inverse function
    flat1 = s[0:(25*(400+1))]
    flat2 = s[(25*(400+1)):]
    omega1 = flat1.reshape((25,401))
    omega2 = flat2.reshape((26,1))
    omega2 = [item for sublist in omega2 for item in sublist]
    omega2 = np.asarray(omega2)
    return omega1,omega2

# Particule class
class Particule():
    def __init__(self,id,m,s,cutoff,c1,c2,w,domain_up,domain_down):
        self.domain_up = domain_up
        self.domain_down = domain_down
        self.omega1 = np.asarray([[random.uniform(self.domain_down,self.domain_up) for j in range(m+1)] for i in range(s)])
        self.omega2 = np.asarray([random.uniform(self.domain_down,self.domain_up) for j in range(s+1)])
        self.sValues = matrixToVector(self.omega1,self.omega2)
        self.vValues = np.zeros(s*(m+1)+s+1)
        self.id = id
        self.bestEnergy = 10000 #Random initial big value
        self.energy = 10000
        self.bValues = 10000 # Random initial value, it will be update by sValues
        self.currentBestbValues = 10000
        self.cutoff = cutoff
        self.c1 = c1
        self.c2 = c2
        self.w = w
    def computeJk(self,k):
        y_k = Y[k]
        x_k = np.append(1,X[k])
        z = sigmoid(np.matmul(self.omega1,x_k))
        z = np.append(1,z)
        h = sigmoid(np.matmul(self.omega2,z))
        return (y_k-h)**2

    def computeJ(self):
        err = 0
        for i in range(len(X)):
            err += self.computeJk(i)
        return err/len(X)

    def update(self):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        self.vValues = self.w*self.vValues + self.c1*r1*(self.bValues-self.sValues) + self.c2*r2*(self.currentBestbValues-self.sValues)

        for i in range(len(self.vValues)): #In case one component is too large or too small we define a maximum value and minimum value of cutoff
            if self.vValues[i] > self.cutoff:
                self.vValues[i] = self.cutoff
            if self.vValues[i] < -self.cutoff:
                self.vValues[i] = -self.cutoff


        self.sValues = self.sValues + self.vValues
        self.omega1, self.omega2 = VectorToMatrix(self.sValues)


def solve(iteration,N,cutoff):
    # Initialize variable
    s = 25
    m = 400
    n = 200
    t_max = iteration
    c1 = 2
    c2 = 2
    w = 0.9
    domain_up = 0.5
    domain_down = - 0.5
    t = 0
    bestGlobalEnergie = [1000]
    frozen = 0 # Count if the best solution dont improve
    diff = 0 # Count if the best solution improve but the improvement is too small
    allBest = []
    Particules = []
    for i in range(N):
        Particules.append(Particule(i, m, s, cutoff,c1,c2,w,domain_up,domain_down))

    while t<t_max and frozen<int(iteration/5) and diff<5: #  Do exactly what the given pseudo code do, exept the frozen statement and the diff statement

        for particule in Particules:
            particule.energy = particule.computeJ()
            if particule.energy <= particule.bestEnergy:
                particule.bestEnergy = particule.energy
                particule.bValues = particule.sValues

        bestEnergies = [particule.bestEnergy for particule in Particules]
        bestEnergieIndex = np.argmin(bestEnergies)
        best = Particules[bestEnergieIndex].bestEnergy
        allBest.append(best)
        if best < bestGlobalEnergie[-1]:
            frozen = 0
            if abs(bestGlobalEnergie[-1]-best)<0.001: #If there is a too much small improvement we increment diff
                diff += 1
            else: #Othwewise we reset it
                diff = 0
            bestGlobalEnergie.append(Particules[bestEnergieIndex].bestEnergy)
            bestGlobalbValue = Particules[bestEnergieIndex].bValues
        else:
            frozen += 1


        for particule in Particules:
            particule.currentBestbValues = bestGlobalbValue
            particule.update()

        t += 1
    omega1,omega2 = VectorToMatrix(bestGlobalbValue)
    bestGlobalEnergie.remove(1000) # Remove the initial random best time
    return omega1,omega2,bestGlobalEnergie,t,allBest # Return the best weight matrices, the list of best energies found, the number of iteration and all best result for each iteration

# print(solve(100,20,5)) #Best parameter found



#Function to compute y predict  (Neural Network with best weighted matrix given)
def computeH(k,omega1,omega2):
    x_k = np.append(1,X[k])
    z = sigmoid(np.matmul(omega1,x_k))
    z = np.append(1,z)
    h = round(sigmoid(np.matmul(omega2,z)))
    return h

def computeYpredict(omega1,omega2):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        y_predict[i] = computeH(i,omega1,omega2)
    return y_predict

def predictionErreur(y_predict):
    err = 0
    for i in range(len(y_predict)):
        if y_predict[i] != Y[i]:
            err += 1
    return err/len(y_predict)



#2) Run 10 times the PSO algorithm and report the optimal value J(Θ1, Θ2)
#you get each time.
bestTimes = []
matrices = []
y_predictedError = []
for i in range(10):
    omega1,omega2,best,t,allBest = solve(100,20,1)
    bestTimes.append(best[-1])
    matrices.append([omega1,omega2])
    y_predictedError.append(predictionErreur(computeYpredict(omega1,omega2)))


print("Voici les 10 résultats obtenues: ",bestTimes)

# 3) For one of these runs, plot J as a function of the iteration
# number.
x = [i for i in range(t)] # x axis for plot energy in function of iteration

plt.plot(x,allBest)
plt.xlabel("Iteration")
plt.ylabel("Energie")
plt.title("Energy as a function of the iteration")
plt.show()

#4 compute predicted error and plot it in function of run of the PSO algorithm
print("Voici un example de y_predict: ",computeYpredict(omega1,omega2))
x = [i for i in range(10)]
plt.plot(x,y_predictedError)
plt.xlabel("Runs")
plt.ylabel("Error")
plt.title("Error as a function over the 10 runs")
plt.show()