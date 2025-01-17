import os,random
file_names = ["hello","hi","hola","namaste"]
os.chdir(f"{os.getcwd()}/practice_1/sample_files")
data=os.listdir()
for i in data:
    name=i.split(".")
    if(len(name)>1):
        file=name[0]
        format=name[1]
        print(file,format)
        os.makedirs(format,exist_ok=True)
        os.rename(f"{os.getcwd()}/{i}",f"{os.getcwd()}/{format}/{i}")

    
    
