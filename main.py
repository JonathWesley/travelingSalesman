import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from City import City
from Fitness import Fitness

# criacao de uma rota
def createRoute(cityList):
    # embaralha uma lista com todas as cidades
    route = random.sample(cityList, len(cityList))
    return route

# cria a populacao inicial
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

# ordena as rotas por maior Fitness
def rankRoutes(population):
    fitnessResults = {}
    # calcula o valor de Fitness de cada rota
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    # retorna esses valores ordenados, com index e valor
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

# seleciona os individuos que vao ser utilizados para gerar a nova geracao
def selection(popRanked, eliteSize):
    selectionResults = []
    # cria um data frame de chave e valor
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    # soma todos os valores de fitness
    df['cum_sum'] = df.Fitness.cumsum()
    # fitness relativa, faz a porcentagem -> 100 * soma dos valores / quantidade de valores
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    # inclui a elite (numero indicado pelo usuario)
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    # percorre o restante do vetor de rank
    for i in range(0, len(popRanked) - eliteSize):
        # selecao por proporcao de Fitness utilizando o metodo da roleta
        # faz com que os individuos com maior Fitness relativa tenham maior probabilidade de serem escolhidos
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

# pega as rotas que serao usadas para reproduzir
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# cruza dois cromossomos para gerar um novo
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    # pega uma quatidade aleatoria de genes do pai1
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    
    # pega o restante dos genes do pai2
    childP2 = [item for item in parent2 if item not in childP1]

    # junta os genes dos dois pais
    child = childP1 + childP2
    return child

# cruza a matingPool para gerar uma nova populacao
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    # adiciona as melhores rotas para a nova geracao
    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    # gera novas rotas a partir de antigas para preencher o restante da populacao
    for i in range(0, length):
        # cruza os pais
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# faz a mutacao de um individuo
def mutate(individual, mutationRate):
    # utiliza a taxa de mutacao como uma probabilidade de trocar 2 cidades em uma rota
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

# faz a mutacao da populacao
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    # percorre toda a populacao mutando ela a partir da taxa de mutacao
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# gera a proxima geracao
def nextGeneration(currentGen, eliteSize, mutationRate):
    # ordena as rotas pelo Fitness
    popRanked = rankRoutes(currentGen)
    # seleciona o index das rotas para cruzar
    selectionResults = selection(popRanked, eliteSize)
    # pega as rotas pela selecao de index
    matingpool = matingPool(currentGen, selectionResults)
    # cruza as rotas e gera uma populacao nova
    children = breedPopulation(matingpool, eliteSize)
    # faz a mutacao da populacao nova
    nextGeneration = mutatePopulation(children, mutationRate)
    # retorna a populacao nova mutada
    return nextGeneration

# executa o algoritmo genetico
def geneticAlgorithm(problemName, population, popSize, eliteSize, mutationRate, generations):
    print("\n\nProblema " + problemName)
    pop = initialPopulation(popSize, population)
    print("Distancia inicial: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Melhor rota inicial: ", end='')
    for x in bestRoute:
        print(str(x.name) + ' ', end='')
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("\nDistancia final: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Melhor rota final: ", end='')
    for x in bestRoute:
        print(str(x.name) + ' ', end='')

# executa o algoritmo genetico plotando o grafico de desenvolvimento
def geneticAlgorithmPlot(problemName, population, popSize, eliteSize, mutationRate, generations):
    print("\n\nProblema " + problemName)
    pop = initialPopulation(popSize, population)
    print("Distancia inicial: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Melhor rota inicial: ", end='')
    for x in bestRoute:
        print(str(x.name) + ' ', end='')
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    print("\nDistancia final: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Melhor rota final: ", end='')
    for x in bestRoute:
        print(str(x.name) + ' ', end='')
    plt.figure('Problema ' + problemName,figsize=(10,8))
    plt.plot(progress)
    plt.title('Problema ' + problemName)
    plt.ylabel('Distancia')
    plt.xlabel('Geracao')
    plt.savefig('Problema' + problemName + '.png', format='png')
    plt.show()

if __name__ == "__main__":
    problemList = []
    problemTam = []
    
    input_file = open("input20.txt")
    for line in input_file.readlines():
        cityNumber = 0
        cityList = []
        data = line.split(';')
        cityNumber = int(data[0])
        iterator = 0
        index = 1
        for i in range(cityNumber-1):
            city = City(iterator)
            for k in range(0, len(cityList)):
                city.connectedTo[k] = cityList[k].connectedTo[city.name]
            for j in range(i+1, cityNumber):
                city.connectedTo[j] = int(data[index])
                index += 1
            cityList.append(city)
            iterator += 1
        city = City(iterator)
        for i in range(len(cityList)):
            city.connectedTo[i] = cityList[i].connectedTo[cityNumber-1]
        cityList.append(city)
        problemList.append(cityList)   
        problemTam.append(cityNumber)
    
    i = 0
    for x in problemList:
        if problemTam[i] <= 5:
            geneticAlgorithmPlot(problemName=str(i), population=x, popSize=10, eliteSize=2, mutationRate=0.01, generations=50)
        elif problemTam[i] > 5 and problemTam[i] <= 10:
            geneticAlgorithmPlot(problemName=str(i), population=x, popSize=50, eliteSize=10, mutationRate=0.01, generations=100)
        else:
            geneticAlgorithmPlot(problemName=str(i), population=x, popSize=100, eliteSize=20, mutationRate=0.01, generations=200)
        i += 1

    