class Animal():
    def __init__(self):
        print("Life has been initiated")
    def eating(self):
        print("Hello I can eat food")
    @staticmethod
    def printRandom(data):
        print(data)


class Human(Animal):
    def __init__(self):
        super().__init__()
        print("Human class has been initiated")
    name="My name is human"
    __age__=28
    def printName(self):
        print(self.name)
    def printAge(self):
        print(self.__age__)
a = Human()
a.printName()
a.printAge()
print(a.__age__)
Animal.printRandom("lkaj;ldfjsdfj")
