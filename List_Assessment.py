#1 Create an empty list called fruits.

# 1. Add "Apple" to the list. 
# 2. Add "Banana" and "Cherry" together. 
# 3. Print the final list. 


fruits=[]
fruits.append("Apple")
fruits.extend(["Banana","cherry"])
print(fruits)

#2 You have a list of colors: 
# colors = ["Red", "Blue", "Green", "Yellow"] 
# 1. Remove "Blue" from the list. 
# 2. Replace "Yellow" with "Black". 
# 3. Print the updated list. 

colors = ["Red", "Blue", "Green", "Yellow"]
colors.remove("Blue")
print(colors)
colors[len(colors)-1]="Black"
print(colors)

# Question 3: Insert at a Specific Position 
# Problem: 
# You are given a list of numbers: 
# numbers = [10, 20, 40, 50] 
# 1. Insert 30 at index 2 (between 20 and 40). 
# 2. Print the final list. 


numbers = [10, 20, 40, 50]
numbers.insert(2,30)
print(numbers)


#  Question 4: Remove and Print Removed Element 
# Problem: 
# You have a list of students: 
# students = ["Alice", "Bob", "Charlie", "David"] 
# 1. Remove "Charlie" from the list and store the removed name in a variable. 
# 2. Print "Charlie has been removed!". 
# 3. Print the updated list. 


students = ["Alice", "Bob", "Charlie", "David"]

removed =  "Charlie" 
students.remove(removed)
print(f"{removed} has been removed from the list")
print(students)



#  Question 5: Slicing Trick 
# Problem: 
# You are given a list of numbers: 
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
# 1. Create a new list even_numbers that contains only even numbers from data using 
# slicing. 
# 2. Print even_numbers.


data = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
even_numbers=data[1::2]
print(even_numbers)


# Bonus Challenge  

#  Problem: 
# You have a list of words: 
words = ["Python", "is", "awesome"] 

a,b,c= words

print(f"{a} {b} {c}")