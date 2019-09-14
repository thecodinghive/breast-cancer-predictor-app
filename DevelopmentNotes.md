
# Local Dev/Env Setup

Run using:

    python3 app.py

Setup:

    # Initial env creation
    conda env create -f environment.yml
    # USe env
    conda activate flask-breast-cancer
    # update env (e.g. after updating deps?)
    conda env update --file environment.yml --prune


# Running on Heroku

After installing new packages, make sure to update `requirements.txt`:

    pip freeze > requirements.txt

The `Procfile` contains the command needed to run the app:

    web: gunicorn app:app --log-file -

Install Heroku:

* Visit https://devcenter.heroku.com/articles/getting-started-with-python#set-up
* Install `git`, if you don't have it already.
* Install `heroku`, following instructions for your platform.

Next, we will need to setup our app on Heroku:

* Create a free Heroku account.
* Create a new application.
* Run `heroku login` locally.
* Follow instructions to deploy: https://dashboard.heroku.com/apps/breastcancerpredictor/deploy/heroku-git

References:

* Heroku `Procfile` guide: https://devcenter.heroku.com/articles/procfile
* https://xcitech.github.io/tutorials/heroku_tutorial/

# Running on CoCalc

* Running server via terminal: https://doc.cocalc.com/howto/webserver.html
* Command: `gunicorn -b 0.0.0.0:5000 app:app --log-file -`
* Then access at: `https://cocalc.com/<project_id>/server/5000/`
* NOTE: Replace `<project_id>` with your project ID, e.g. `https://cocalc.com/cfca8905-cec9-4b44-b37e-33f41dcc8068/server/5000/`. You can get this from the URL when browsing the project in cocalc.



