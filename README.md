#Source article : https://mlflow.org/docs/latest/getting-started/quickstart-2/index.html

Steps to run the python file 'to_compare_runs_wine.py'

Here using VSC command prompt.

1. Create an virtual env in your workfolder


2. Inside(active) the env, install requirements.txt
$pip install -r requirements.txt

3. Check it's installed correctly.
use these commands to check the version : a) $mlflow --version
                                          b) $pip show tensorflow

4.Run the file,
$python to_compare_runs_wine.py

5.If we want to check the UI side by side , then open another command prompt, actiavte venv.
run : $mlflow ui

This will show line like this  : INFO:waitress:Serving on http://127.0.0.1:5000
Click on that link, that will open in the browser

Don't close this command propmt(if closed, then the page can't be reach)

