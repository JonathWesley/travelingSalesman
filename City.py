class City:
    def __init__(self, name):
        self.name = name
        self.connectedTo = {}
    
    def distance(self, city):
        distance = self.connectedTo[city.name]
        return distance
    
    def __repr__(self):
        string = str(self.name) + ": ("
        for key, value in self.connectedTo.items():
            string += str(key) + '->' + str(value) +';'
        string += ')'
        return string
            
# apenas para teste da classe City
if __name__ == "__main__":
    cityA = City('A')
    cityA.connectedTo['B'] = 2
    cityA.connectedTo['C'] = 5
    
    cityB = City('B')
    cityB.connectedTo['A'] = 2
    cityB.connectedTo['C'] = 5
    
    print(cityA)
    print(cityA.distance(cityB))