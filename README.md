# Predizione della malattia cardiovascolare con modelli di Machine Learning

Questo progetto affronta la classificazione del rischio cardiovascolare a partire da dati clinici, attraverso l’utilizzo e la comparazione di diversi modelli di machine learning supervisionati. Il lavoro è stato svolto come parte del corso universitario di Machine Learning.

##  Contenuto della repository

- `01_naivebayes_qda_lasso.R`: Naive Bayes (normale e kernel), QDA e regressione Lasso.
- `02_decision_tree_random_forest.R`: modelli ad albero e random forest, con tuning e analisi ROC.
- `03_neuralnet_knn_pls.R`: Reti neurali (perceptron e MLP), KNN, PLS con tuning e validazione.
- `cardio_data_processed.csv`: dataset principale preprocessato.
- `train.csv` & `test.csv`: subset per analisi.
- `SOMMARIO_MODELLI.pdf`: confronto finale tra performance dei modelli (AUC, F1-score, Specificità, Sensibilità).

##  Obiettivo

Costruire un sistema di classificazione del rischio cardiovascolare a partire da variabili cliniche (pressione arteriosa, BMI, colesterolo, età, stile di vita). I modelli esplorati includono:

- QDA (Quadratic Discriminant Analysis)
- Decision Tree e Random Forest
- Naive Bayes (con distribuzione normale e KDE)
- Logistic Lasso
- Reti Neurali (Perceptron, MLP)
- K-Nearest Neighbors (KNN)
- Partial Least Squares (PLS)

##  Dataset

Il dataset contiene circa 70.000 osservazioni ed è stato sottoposto a:
- bilanciamento classi (downsampling)
- selezione di variabili rilevanti
- gestione outlier e collinearità
- tuning personalizzato delle soglie di classificazione per AUC e F1-score

##  Pacchetti principali usati

- `caret`, `pROC`, `nnet`, `randomForest`, `MASS`, `e1071`, `NeuralNetTools`, `rpart`, `ggplot2`, `klaR`
- Funzioni custom per tuning su metriche diverse (Accuracy, Sensibilità, Specificità, AUC)

##  Autori

Nicolò Bachiorri, Emma Barrow, Chiara Mezzanzanica, Emanuele Saccardo
