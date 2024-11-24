# NLP-Group-3-Final-Project
NLP Group 3 Final Project


## Folders
### Preprocessing
Brian's preprocessing of political texts
Preprocessing/data - a collection of raw data
Preprocessing/finalized-data - a collection of finalized usable data for modeling

### Sources
* https://www.congress.gov/
* https://www.govinfo.gov/
* https://www.presidency.ucsb.edu/

### Steps to run preprocessing
1. Virtual env setup (for mac)
```
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```
2. Install requirements
```
pip freeze > requirements.txt
pip install -r requirements.txt
```
3. Run text-combiner.py to combine the congress texts into articles.csv
4. Run main.py which processes the text and cleans it into a usable format for the models

### Model 1
...

### Model 2
...
