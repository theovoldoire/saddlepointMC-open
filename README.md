## Organization of the repo

The repository is an extraction from the initial (private) repository used for the project. Scripts used in the project are available in src/. Result of the inference files on voting data are available in results/. 

Description of the files: 

- "functions.py": main interest for people wanting to understand our implementation of Saddlepoint Monte Carlo, with documentation.
- "main_study.ipynb": interactive notebook to run the different studies on synthetic and real data.
- "primitives.py": two functions used in functions.py but not relevant for saddlepoint Monte Carlo. 
- "data_preparation.R": takes as input the data files downloaded on the Interior Ministry website for election results, and transforms them into workable data. Also merge with context variables for different elections and documents the different cases for the 2024 legislative elections.
- "figures_crafter.R": used on the generated results files to craft tables included in the article.
