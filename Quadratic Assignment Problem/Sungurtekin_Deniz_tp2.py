import numpy as np
import random

# Déclaration des variables nécessaire à la résolution de QAP

N = 12 # Nombre de locations
L = [int(1*N),int(0.5*N),int(0.9*N)] # la taille de la liste tabu, un élément est suprimé après 6 itérations (tabu tenure)
nombreVoisins = int(N * (N-1) / 2) # neighboorhood : n*(n-1) / 2, with N = 12 -> 66
iterations = 20000

# Declaration de nos matrices distance et flow
D = np.matrix('0 1 2 3 1 2 3 4 2 3 4 5;1 0 1 2 2 1 2 3 3 2 3 4;2 1 0 1 3 2 1 2 4 3 2 3;3 2 1 0 4 3 2 1 5 4 3 2;1 2 3 4 0 1 2 3 1 2 3 4;2 1 2 3 1 0 1 2 2 1 2 3;3 2 1 2 2 1 0 1 3 2 1 2;4 3 2 1 3 2 1 0 4 3 2 1;2 3 4 5 1 2 3 4 0 1 2 3;3 2 3 4 2 1 2 3 1 0 1 2;4 3 2 3 3 2 1 2 2 1 0 1;5 4 3 2 4 3 2 1 3 2 1 0')
W = np.matrix('0 5 2 4 1 0 0 6 2 1 1 1;5 0 3 0 2 2 2 0 4 5 0 0;2 3 0 0 0 0 0 5 5 2 2 2;4 0 0 0 5 2 2 10 0 0 5 5;1 2 0 5 0 10 0 0 0 5 1 1;0 2 0 2 10 0 5 1 1 5 4 0;0 2 0 2 0 5 0 10 5 2 3 3;6 0 5 10 0 1 10 0 0 0 5 0;2 4 5 0 0 1 5 0 0 0 10 10;1 5 2 0 5 5 2 0 0 0 5 0;1 0 2 5 1 4 3 5 10 5 0 2;1 0 2 5 1 0 3 0 10 0 2 0')

D = D.tolist()
W = W.tolist()

# Initialisation de variable contenant les voisins possibles
neighbors = np.zeros((nombreVoisins, N +2), dtype=int)

# Fonction calculant la fitness d'une configuration
def cout(sol):
  cost=0
  for i in range(N):
    for j in range(N):
        cost+=D[i][j] *W[sol[i]][sol[j]]
  return cost

#Verifie si une configuration n'est pas présente dans la tabu list
def notInTabu (solution, tabu):
    notFound = False
    if not solution.tolist() in tabu:
        solution[0], solution[1] = solution[1], solution[0]
        if not solution.tolist() in tabu:
            notFound = True

    return notFound

#Effectue l'interchangement des positions
def permute(solution):

    global idx, neighbors
    for i in range (N):
        j=i+1
        for j in range(N):
            if  i<j:
                idx=idx+1
                solution[j], solution[i] = solution[i], solution[j]
                neighbors[idx, :-2] = solution
                neighbors[idx, -2:] = [solution[i], solution[j]]
                solution[i], solution[j] = solution[j], solution[i]

#Resolution du problème
def solveQAP(L,iterations):
    global neighbors, idx
    currentSol = random.sample(range(N), N) #Génere aléatoirement une configuration
    bestSol = currentSol
    Tabu = []
    frequency = {}

    while iterations > 0:

        idx = -1
        permute(currentSol)

        cost = np.zeros((len(neighbors)))  # Contiendra le cout des voisins
        for index in range(len(neighbors)):
            cost[index] = cout(neighbors[index, :-2])  # evalue le cout des voisins
        rang = np.argsort(cost)  # On trie par cout en retournant les indexes
        neighbors = neighbors[rang] # On obtient les meilleurs voisins

        for j in range(nombreVoisins):

            not_tabu = notInTabu(neighbors[j, -2:], Tabu)
            if (not_tabu ):
                currentSol = neighbors[j, :-2].tolist()
                Tabu.append(neighbors[j, -2:].tolist())

                if len(Tabu) > L-1:
                    Tabu = Tabu[1:]

                #frequences
                if not tuple(currentSol) in frequency.keys():
                    frequency[tuple(currentSol)] = 1

                    if cout(currentSol) <  cout(bestSol):
                        bestSol = currentSol
                else:

                    coutCourant= cout(currentSol) + frequency[tuple(currentSol)] # On pénalise selon la fréquence
                    frequency[tuple(currentSol)] += 1   # On incrémente la fréquence de la configuration courante

                    if coutCourant <  cout(bestSol):
                        bestSol = currentSol

                break

            #Aspiration

            elif cout(neighbors[j, :-2]) <  cout(bestSol):

                currentSol = neighbors[j, :-2].tolist()

                Tabu.insert(0, Tabu.pop(Tabu.index(neighbors[j, -2:].tolist())))


                if len(Tabu) > L - 1:
                    Tabu = Tabu[1:]

                    # frequence
                if not tuple(currentSol) in frequency.keys():
                    frequency[tuple(currentSol)] = 1  # set key->penality -> to One
                    bestSol = currentSol

                else:

                    coutCourant= cout(currentSol) + frequency[tuple(currentSol)] # ajoute le poids de la fréquence
                    frequency[tuple(currentSol)] += 1   # increament la fréquence correspondante

                    if coutCourant <  cout(bestSol):
                        bestSol = currentSol

        iterations -= 1


    return [bestSol,cout(bestSol)]

# Stocke et affiche le meilleur fitness, la moyenne et l'écart type des 10 itérations avec chacunes des valeurs de L
bestSolutionsL1 = []
bestSolutionsL2 = []
bestSolutionsL3 = []

for i in range(10):
    bestSolutionsL1.append(solveQAP(L[0],iterations)[1])

for i in range(10):
    bestSolutionsL2.append(solveQAP(L[1],iterations)[1])

for i in range(10):
    bestSolutionsL3.append(solveQAP(L[2],iterations)[1])

bestSolutionsL1 = np.asarray(bestSolutionsL1)
bestSolutionsL2 = np.asarray(bestSolutionsL2)
bestSolutionsL3 = np.asarray(bestSolutionsL3)

bestValueL1 = np.amin(bestSolutionsL1)
bestValueL2 = np.amin(bestSolutionsL2)
bestValueL3 = np.amin(bestSolutionsL3)
best = [bestValueL1]+[bestValueL2]+[bestValueL3]

meanL1 = np.mean(bestSolutionsL1)
meanL2 = np.mean(bestSolutionsL2)
meanL3 = np.mean(bestSolutionsL3)
mean = [meanL1]+[meanL2]+[meanL3]

stdL1 = np.std(bestSolutionsL1)
stdL2 = np.std(bestSolutionsL2)
stdL3 = np.std(bestSolutionsL3)
std = [stdL1]+[stdL2]+[stdL3]

for i in range(len(L)):
    print("Pour L = %d: le meilleur solution est %d, on a une moyenne de %f et un std de %f" %(L[i],best[i],mean[i],std[i]))