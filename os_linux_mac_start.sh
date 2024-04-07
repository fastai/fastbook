python3 -m venv env

source env/bin/activate

python3 -m pip --upgrade pip 

pip install -r requirements.txt

jupyter notebook --no-browser
