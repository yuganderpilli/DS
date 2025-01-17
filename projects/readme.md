File Organizer
This Python script helps to move files into respective folders based on their file format. It reads the files in the sample_files directory, organizes them by their extensions, and moves them into separate folders named after the file formats.

Features
Automatically sorts files based on their extensions.
Creates folders named after the file formats (e.g., .txt files will be moved into a folder named txt).
The script ensures that existing folders for each format are not overwritten.
How It Works
The script first changes the working directory to the sample_files folder inside the practice_1 directory.
It lists all files in the directory.
For each file:
The file name is split into its base name and format (extension).
If the file has an extension, it is moved into a folder named after its extension.
If the folder for a specific extension doesn't already exist, it is created.
Files are renamed and moved into their respective format-based folders.
Prerequisites
Python 3.x
The os module (which is built-in)
Directory Structure
Ensure the directory structure looks like this:

markdown
Copy
Edit
practice_1/
    sample_files/
        hello.txt
        hi.pdf
        hola.jpg
        namaste.txt
Running the Script
Place your files in the sample_files directory inside the practice_1 folder.
Run the script, and the files will be sorted into the appropriate format-based folders.
Example Output
If the script processes the files:

Copy
Edit
hello txt
hi pdf
hola jpg
namaste txt
It will create the following folder structure:

markdown
Copy
Edit
practice_1/
    sample_files/
        txt/
            hello.txt
            namaste.txt
        pdf/
            hi.pdf
        jpg/
            hola.jpg
License
This project is licensed under the MIT License.