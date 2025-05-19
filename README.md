Steps

D:\projects\mlopsProject>python -m venv myenv

D:\projects\mlopsProject>myenv\Scripts\activate

(myenv) D:\projects\mlopsProject>git init
Initialized empty Git repository in D:/projects/mlopsProject/git/

(myenv) D:\projects\mlopsProject>git add README.md

(myenv) D:\projects\mlopsProject>git commit -m "first commit"

(myenv) D:\projects\mlopsProject>git status

(myenv) D:\projects\mlopsProject>git branch -M main

(myenv) D:\projects\mlopsProject>git remote add origin https://github.com/PiyushVIT346/mlopsProject.git

(myenv) D:\projects\mlopsProject>git remote -v
origin  https://github.com/PiyushVIT346/mlopsProject.git (fetch)
origin  https://github.com/PiyushVIT346/mlopsProject.git (push)

(myenv) D:\projects\mlopsProject>git push -u origin main
create new folder setup.py and src and file __init__.py in src folder. Write the code in setup.py. 
(myenv) D:\projects\mlopsProject>pip install -r requirements.txt
git status
git add .
git commit -m "setup.py"
git push origin main

create folder components


(myenv) D:\projects\mlopsProject>python src/logger.py

to run complete project : python src/components/data_ingestion.py
