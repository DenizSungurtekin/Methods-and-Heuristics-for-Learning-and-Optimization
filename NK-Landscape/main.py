import math
import random
import copy
import matplotlib.pyplot as plt
# Definitions des fonctions qui permette le calcule de F:
def computef0(x):
    if x=="0":
        return 2
    if x=="1":
        return 1
    else:
        print("Erreur de input f0")

def computef1(x):
    if x=="00":
        return 2
    if x=="01":
        return 3
    if x=="10":
        return 2
    if x=="11":
        return 0
    else:
        print("Erreur de input f1")

def computef2(x):
    if x=="000":
        return 0
    if x=="001":
        return 1
    if x=="010":
        return 1
    if x=="011":
        return 0
    if x=="100":
        return 2
    if x=="101":
        return 0
    if x=="110":
        return 0
    if x=="111":
        return 0
    else:
        print("Erreur de input f2")

# Calcule de Fk
def computeF(x,K):
    if K == 0:
        outputs = []
        for element in x:
            outputs.append(computef0(element))
        result = sum(outputs)
        return result
    if K == 1:
        outputs = []
        for i in range((len(x)-K)):
            outputs.append(computef1(x[i]+x[i+1]))
        result = sum(outputs)
        return result
    if K == 2:
        outputs = []
        for i in range((len(x)-K)):
            outputs.append(computef2(x[i]+x[i+1]+x[i+2]))
        result = sum(outputs)
        return result

# Fonction permettant de generer une sequence de bits de longueur N
def GenerateRandomSequence(N):
    randomSequence = []
    for i in range(N):
        x = random.randint(0,1)
        randomSequence.append(x)

    Sequence = ""

    for element in randomSequence:
        Sequence += str(element)
    return Sequence

# Fonction permettant de convertir une liste en un seul string (Mes sequences sont des strings donc il est nécessaire de les convertirs en liste pour modifier leur contenu et de revenir a l'état initial.)
def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
    return str1


# Fonction calculant les voisins d'une séquence, c-a-d l'ensemble de séquence différant de un seul bit
def neighbours(x,N):
    x = list(x)
    neighbours = []
    for i in range(N):
        if x[i] == "0":
            x[i] = "1"
            neighbours.append(listToString(x))
            x[i] = "0"

        if x[i] == "1":
            x[i] = "0"
            neighbours.append(listToString(x))
            x[i] = "1"

    return neighbours

# Deterministic Hill-Climbing
def DeterministicHill(K,N):
    #Definit la séquence initial
    currentNode = GenerateRandomSequence(N)
    print("Starting Node: ",currentNode)
    # On calcule le cout de chacun des voisin, prend le maximum et on s'arrete lorsque le cout des voisins ne depasse pas celui du noeud courant.
    while True:
        bool = False
        currentCost = computeF(currentNode,K)
        voisins = neighbours(currentNode,N)
        for element in voisins:
            cout = computeF(element,K)
            if cout > currentCost:
                cout = currentCost
                currentNode = element
                bool = True
        if not bool:
            return [currentNode,computeF(currentNode,K)]

# Probabilistic Hill-Climbing
# On definit la fonction qui calcule la probabilite donne dans l'enonce du TP
def computeProbability(x,K,V):
    cout = computeF(x,K)
    coutAllVoisins = []
    for element in V:
        coutAllVoisins.append(computeF(element,K))
    sommeCout = sum(coutAllVoisins)
    return cout/sommeCout

# On s'assure de prendre le noeud avec la meilleur fitness en prenant celle qui a la plus grande probabilite
def ProbabilisticHill(K,N):
    # Meme raisonnement que la methode deterministe où à chaque iteration on prend le voisin avec la meilleur probabilite
    currentNode = GenerateRandomSequence(N)
    print("Starting Node: ",currentNode)

    for i in range(100):
        voisins = neighbours(currentNode,N)
        currentProbability = computeProbability(currentNode, K,voisins)
        for element in voisins:
            probabilitie = computeProbability(element,K,voisins)
            if probabilitie > currentProbability:
                probabilitie = currentProbability
                currentNode = element
    return [currentNode,computeF(currentNode,K)]


#No Error handling
K = input("Entrer une valeur pour K (Entre 0 et 2)")
# Enter 1 for deterministic and 2 for probabilistic
methode = input("Entrer la methode voulue (1 ou 2)")
if methode == "1":
    print("Final Node: ",DeterministicHill(int(K),21))
if methode == "2":
    print("Final Node: ",ProbabilisticHill(int(K),21))





