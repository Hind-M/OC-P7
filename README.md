# OC-P7

The notebook "P7.ipynb" contains the cleaning, analysis and modeling based on the provided dataset in order to predict loan payment failure of the clients.

The files "Procfile", "requirements.txt" and "runtime.txt" are intended for Heroku configuration in order to deploy the dashboard application.

The "requirements.txt" file also gives the user information about the packages needed to run the dashboard application locally.

The "app_dashboard.py" is the dashboard application code. It loads the needed data from the folder with the same name.

The dashboard is available under the following url:

https://loanattribution-dashboard.herokuapp.com/

PS:

Heroku put their apps on sleep mode if there is only one web dyno and that dyno doesn't receive any traffic in 1 hour.
When someone accesses the app, the dyno manager will automatically wake up the web dyno to run the web process type. This causes a short delay for this first request, but subsequent requests will perform normally.

So please be patient by giving it some time to wake up.