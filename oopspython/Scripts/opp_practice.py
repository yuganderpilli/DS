class Animal:
    def __init__(self):
        print("I am living being")
class Human(Animal):
    def __init__(self,name):
        super().__init__()
        self.actual_name=name
        print("I am a human being")
    @property
    def name(self):
        return self.actual_name
    @name.setter
    def name(self,value):
        self.actual_name=value
    


jathin=Human("jathin naga sai")
jathin.name="jathin nagasai"
print(jathin.name)

print(dir(Human))