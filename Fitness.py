class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    
# apenas para teste da classe Fitness
if __name__ == "__main__":
    from City import City
    
    cityList = []
    
    city = City(0)
    city.connectedTo[1] = 1
    city.connectedTo[2] = 1
    cityList.append(city)
    
    city = City(1)
    city.connectedTo[0] = 1
    city.connectedTo[2] = 1
    cityList.append(city)
    
    city = City(2)
    city.connectedTo[0] = 1
    city.connectedTo[1] = 1
    cityList.append(city)
    
    route = [cityList[1], cityList[0], cityList[2]]
    
    print(Fitness(route).routeFitness())