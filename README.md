# Predicting-Comprehensive-HIV-AIDS-Knowledge-Using-HIV-Awareness-Testing-and-Information-in-Uganda
This project investigates whether information related to HIV awareness, HIV testing, HIV information sources, and knowledge of HIV transmission can be used to predict comprehensive HIV/AIDS knowledge in Uganda.
The project uses secondary data from world bank and applies data-processing, exploratory data-analysis, and machine-learning techniques to identify patterns associated with comprehensive HIV/AIDS knowledge.

#Research question
Do HIV testing uptake, information exposure and awareness jointly explain comprehensive HIV/AIDS knowledge, viewed through a Knowledge-Attitudes-Practices (KAP) framework?

#Main Objective
To develop and evaluate machine-learning models for predicting comprehensive HIV/AIDS knowledge in Uganda using selected HIV awareness, testing, information, and transmission-knowledge indicators.

#Project Indicators

The analysis uses selected indicators related to:

Percentage distribution of women who have been tested for the AIDS virus.
Percentage of men who think that discussion of AIDS in the media is not acceptable in any media.
Percentage of men with comprehensive knowledge about AIDS.
Percentage of individuals who have heard of AIDS.
Percentage of individuals who have received information about AIDS from specific sources.
Percentage of men who know a source for the HIV test.

#Technologies Used

Python
Pandas
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook
VS Code
Git and GitHub

#Repository Structure

Create project
     ↓
Create .venv
     ↓
Create .gitignore
     ↓
Create README.md
     ↓
Create requirements.txt
     ↓
Create data/notebook/src/test folders
     ↓
git add .
     ↓
git commit
     ↓
git push
     ↓
GitHub

#Running the project

#create a virtual environment

open my folder: HIV Awareness
created a virtual environment named .venv using Python's built-in venv module. In the VS Code terminal, run: python -m venv .venv

#installing dependency packages

installed the packages to be used in my project: python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter

#defining my functions in python

created data_processing.py under the scr folder
ensured to import the packages to be used in our project such as numpy and pandas
defined the functions that I used to clean my data. 
defined the functions that I used to group my data for example grouping the ages in the data.
defined the Indicator variable in my data. 
I also created _init_.py and left it empty. 

#Openning the Jupyter notebook and running the VS. Code

created notebooks/HIV_Awareness.ipynb
imported the defined functions under data_processing.py to be used for my project
ensured that the parameter of the function imported my csv data for analysis.
ensured the data was cleaned and the ages were grouped well.
Ran All

#Running the second Jupyter notebook and running the VS. Code

imported the package to used for data visualization.
defined the functions to be used.
Ran All

#Running the third Jupyter notebook and running the VS. Code

defined the six relevant HIV/AIDS indicators from my dataset.
Converted the data from long format to wide format so that each indicator could be used as a separate machine-learning variable.
Renamed the indicators with shorter, more interpretable variable names.
The dataset was divided into training (80%) and testing (20%) subsets.
A fixed random_state=42 was used to make the train-test split reproducible.
Ran All

# Running the Jupyter notebooks

Restart and Run All the jupyter notebooks and ensure that they donot have errors.

#Interpretation

The findings suggest that indicators related to HIV awareness, testing, access to information, and knowledge of HIV transmission may provide useful information for predicting comprehensive HIV/AIDS knowledge among men.
However, the machine-learning results should be interpreted as predictive associations rather than evidence that one indicator directly causes comprehensive HIV/AIDS knowledge. Finaly, AI was used to correct specific codes that failed to run when carrying out this project.

#Limitations

The study has several limitations:

The analysis uses secondary data rather than primary individual-level data collected specifically for this research.
The ages weren't categorized appropriately for data analysis. 
The available indicators may not capture all factors associated with HIV/AIDS knowledge.
The quality of predictions depends on the quality and completeness of the available data.
Associations identified by machine-learning models should not automatically be interpreted as causal relationships.
If the dataset is aggregated, predictions may not represent individual men's characteristics.
Model performance may change when applied to a different population or dataset.







