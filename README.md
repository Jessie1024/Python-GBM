# bios611-project3

The goal of the project is to use tree-based machine learning method to predict the probability of heart disease by patients clinical information. The prediction will be helpful for the patients to get prepare for their potential life-threatening heart disease.


## data
The data comes from the Heart Disease Cleveland UCI, can be accessed from here: https://www.kaggle.com/cherngs/heart-disease-cleveland-uci The dataset including 303 the patients, with 108 cases had no heart disease and 165 cases have experience at least one heart disease before. Clinical information include: age, sex, chest pain type, resting blood pressure, serum cholesterol level, fasting blood sugar, resting EKG result, maximum heart rate, exercise induced angina, ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels colored by flourosopy, thal, and if has experienced the heart disease before.

The description data can be found here: https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877

## prediction by machine learning methods
In this project, we are interested which tree-based method is better predict the hear disease-the random forest(RF) for the gradient boost machine (GBM) the assessment is conducted by ROC curve and the precision score.

## how to run the app on docker
first, build the docker environment by:

    >docker build . -t project3
    
Then, run the R enviornment by :
    >docker run -v /home/"YOUR DIRECTORY"/"YOUR PROJECT":/home/rstudio -p 8787:8787 -e PASSWORD=mypassword -i project3

Or, run the jypyter notebook environment by: 
    >docker run -p 8765:8765 -v /home//home/"YOUR DIRECTORY"/"YOUR PROJECT":/home/rstudio -e PASSWORD=mypassword -it project3 sudo -H -u rstudio /bin/bash -c "cd ~/;jupyter lab  --ip 0.0.0.0 --port 8765"'
    
To make the final report html, say:
    >make clean
    >make Report.html
To make the assessment curve, say:
    >make clean
    >make full_figure.pdf
