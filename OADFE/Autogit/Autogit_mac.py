"""
Made by burunduk387
Tested by FacPhys
!!!REQUIRES GIT & GITPYTHON INSTALLATION!!!
"""

import time
import numpy as np
from git import Repo

def initrepo():
    username = input("Your GitHub username:\n")
    surname = input("Your surname:\n")
    name = input("Your name:\n")
    email = input("Your email:\n")
    ans = input("Would you like to enable autohttps? y or n:\n")
    if (ans == "y"):
        phoning = 1
    elif (ans == "n"):
        phoning = 0
    else:
        print("Are you ok there?\n")
        initrepo()
    ans = input("Does your repo exist locally? y or n:\n")
    if (ans == "y"):
        repadr = input("What is the name?\n")
        try:
            repo = Repo(repadr)
        except:
            print("Are you ok there?\n")
            initrepo()
    elif (ans == "n"):
        rephttps = input("Give me the https for cloning:\n")
        try:
            path = "git-exam-" + username
            repo = Repo.clone_from(rephttps, path)
        except:
            print("Are you ok there?\n")
            initrepo()
    else:
        print("Are you ok there?\n")
        initrepo()
    config = repo.config_writer()
    config.set_value("user", "email", email)
    config.set_value("user", "name", name + " " + surname)
    return(username, surname, name, email, repo, phoning)
def resolve(filepath):
    with open(filepath, "r") as f:
        got = set(f.readlines())
        gave = sorted(got)
    with open(filepath, "w") as f:
        for i in gave:
            if "@" in i:
                f.write(i.rstrip("\n") + "\n")

username, surname, name, email, repo, phoning = initrepo()
print("Great, we initiated!\n")

path = str(repo.working_tree_dir)
filepath = path + "/students.txt"
git = repo.git

origin = repo.remotes[0].url
origin = origin[19:]
password = input("Your GitHub password:\n")
origin = "https://" + username + ":" + password + "@github.com/" + origin

ans = input("Was your name already commited? y or n\n")
if (ans == "n"):
    with open(filepath, "a") as f:
        f.write("\n")
        f.write(name + " " + surname + " <" + email + ">")
    git.add(".")
    ПОМЕНЯЙ ЭТО СНИЗУ НА СВОЕ СООБЩЕНИЕ ВМЕСТО АДЕД МАЙ НЕЙМ
    git.commit("-m", "Added my name")
    git.push(origin)
    
if (phoning == 0):
    fetchers = []
    k = input("Now give me fetch data line by line, with nn when you're done:\n")
    while(k != "nn"):
        fetchers.append(k)
        k = input()
if (phoning == 1):
    import urllib
    url = "https://raw.githubusercontent.com/burunduk387/Autogit/main/fetchers.txt"
    f = urllib.request.urlopen(url)
    fetchers = [x.decode("utf-8").rstrip("\r\n") for x in f]
    
rng = np.random.default_rng()
rng.shuffle(fetchers)
print("Ready for takeoff\n")

try:
    while (True):
        for i in fetchers:
            time.sleep(np.random.randint(110, 190))
            try:
                git.fetch(i)
                git.merge("--allow-unrelated-histories", "FETCH_HEAD")
            except:
                resolve(filepath)
                time.sleep(np.random.randint(6, 12))
                git.add(".")
                ПОМЕНЯЙ ЭТО СНИЗУ НА СВОЕ СООБЩЕНИЕ
                В ДВОЙНЫХ КАВЫЧКАХ ПРОСТО
                message = "Merge " + i
                git.commit("-m", message)
                git.push(origin)
            
except KeyboardInterrupt:
    pass