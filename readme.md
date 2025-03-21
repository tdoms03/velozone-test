# Velozone

## Installation
(pip) Install the required packages available in [requirements.txt](requirements.txt)

## Usage
Run the following command in the terminal. This starts a webpage via http://127.0.0.1:5000, which is a development server.

    python main.py

## UPDATES website
[main.py](main.py) contains all website code for routing (fuse of [app.py](app.py) and application routing code i created)<br>
website contains 3 html scripts:
1. login
2. website dashboard
3. base

folder structure should be:
- main.py and data_analysis.py in main folder
- static folder with img, css and js (containing gamification.js and script.js) folder
- instance folder with users.db
- frontend folder with Website_V1.0.html, login.html and base.html<br>
![image](https://github.ugent.be/audlbeke/Sport_Gand_Adaptive/assets/18048/19594795-e906-4eb6-9969-5b74d2770633)


## Deployment to the world wide web
[link](https://medium.com/@mutabletechke/step-by-step-guide-deploying-a-flask-application-to-a-linux-server-e98b0be68ce6) naar site die 'eenvoudig' uitlegt hoe je een flask app kunt deployen op een linux server

## Github usage
### Pulling Code
To get the latest code from github, run the following command:

    git pull

### Pushing Code
After making changes to the code, you can commit the changes. A commit makes a local 'checkpoint':

    git add .
    git commit -m 'some message'

When a commit is made, it is still in your local repository. You don't necessarily need to perform a push after each commit. You can either choose to keep on coding or to push the commit to the remote repository (on github).

To push the code:

    git push
