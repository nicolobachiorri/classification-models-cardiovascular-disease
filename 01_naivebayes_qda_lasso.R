setwd("/Users/nicolobachiorri/Desktop/PROGETTO ML ")

#Cardiovascular Disease. predict the presence or absence of cardiovascular disease.
r.tot <- read.csv("cardio_data_processed.csv", sep=",", header = T, stringsAsFactors = T, na.strings=c("NA","NaN", "","Unknown"))

#controllo variabili costanti
apply(r.tot, 2, function(x) length(unique(x)) == 1)

#controllo missing value
sapply(r.tot, function(x)(sum(is.na(x))))


#### preparazione dataset ####

#riclassificazione variabili ed eliminazione variabili inutili
str(r.tot)
r.tot$gender <- as.factor(r.tot$gender)
r.tot$cholesterol <- as.factor(r.tot$cholesterol)
r.tot$gluc <- as.factor(r.tot$gluc)
r.tot$smoke <- as.factor(r.tot$smoke)
r.tot$alco <- as.factor(r.tot$alco)
r.tot$active <- as.factor(r.tot$active)
r.tot$cardio <- as.factor(r.tot$cardio)
str(r.tot)

summary(r.tot)
table(r.tot$cardio)/nrow(r.tot)   #è bilanciato (le prior vere sono 1=0.75 0=0.25)

cardio_1 <- subset(r.tot, cardio == 1)  # Estrai le osservazioni con cardio = 1
cardio_0 <- subset(r.tot, cardio == 0)  # Estrai le osservazioni con cardio = 0

set.seed(1)
sample_cardio_0 <- cardio_0[sample(nrow(cardio_0), 30000), ]  # Campiona 30000 osservazioni da cardio = 0
sample_cardio_1 <- cardio_1[sample(nrow(cardio_1), 10000), ]  # Campiona 10000 osservazioni da cardio = 1

r <- rbind(sample_cardio_1, sample_cardio_0)
anyDuplicated(r)   #confermo di nn aver preso due volte la stessa osservazione
r <- r[sample(nrow(r)), ]   #se voglio posso mescolare le osservazioni in r

table(r$cardio)/nrow(r)   #ora le prior sono corrette

str(r)
r<-r[,-c(1,2)]   #elimino id e age


#### divisione in dataset di training e validation ####

set.seed(1)  
train.index.r <- sample(c(1:dim(r)[1]), dim(r)[1]*0.6)  
train <- r[train.index.r, ]
test <- r[-train.index.r, ]
table(train$cardio)/nrow(train)
table(test$cardio)/nrow(test)

#write.csv(train, "train.csv", row.names = FALSE) 
#write.csv(test, "test.csv", row.names = FALSE) 





#### caricamento pacchetti necessari #### 

# Installazione pacchetti necessari (se non già installati)
install.packages(c("e1071", "klaR", "ggplot2", "caret", "dplyr", "PerformanceAnalytics", "funModeling", "Hmisc"))

# Caricamento librerie
library(e1071)
library(klaR)
library(ggplot2)
library(caret)
library(dplyr)
library(PerformanceAnalytics)
library(funModeling)
library(Hmisc)

# Caricamento dataset
train_df <- read.csv("train.csv")
test_df <- read.csv("test.csv")

# Struttura del dataset
str(train_df)
str(test_df)


#### Conversione delle variabili categoriche in fattoriin train_df ####
train_df$gender <- as.factor(train_df$gender)
train_df$cholesterol <- as.factor(train_df$cholesterol)
train_df$gluc <- as.factor(train_df$gluc)
train_df$smoke <- as.factor(train_df$smoke)
train_df$alco <- as.factor(train_df$alco)
train_df$active <- as.factor(train_df$active)

# Conversione della variabile target in fattore
train_df$cardio <- as.factor(train_df$cardio)


#### Conversione delle variabili categoriche in fattori in test_df ####
test_df$gender <- as.factor(test_df$gender)
test_df$cholesterol <- as.factor(test_df$cholesterol)
test_df$gluc <- as.factor(test_df$gluc)
test_df$smoke <- as.factor(test_df$smoke)
test_df$alco <- as.factor(test_df$alco)
test_df$active <- as.factor(test_df$active)

# Conversione della variabile target in fattore in test_df
test_df$cardio <- as.factor(test_df$cardio)

#### ZERO VARIANCE ####
library(caret)
nzv <- nearZeroVar(train_df, saveMetrics = TRUE)
print(nzv) #perfetto

#### ANALISI DELLA COLLINEARITA' ####

# Selezioniamo solo variabili numeriche
numeric_vars <- select(train_df, where(is.numeric))

# Matrice di correlazione
corr_matrix <- cor(numeric_vars)

# Visualizzazione delle correlazioni
#install.packages("corrplot")  # Installa il pacchetto (se non lo hai già)
library(corrplot)  # Carica il pacchetto

corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.7)

# Identificazione variabili altamente correlate (cutoff > 0.90)
correlatedPredictors <- findCorrelation(cor(numeric_vars), cutoff = 0.90, names = TRUE)
print(correlatedPredictors) 

#### GESTIONE MISSING VAlUES #### 

# Percentuale di NA per ogni variabile in train_df
sapply(train_df, function(x) sum(is.na(x)) / length(x) * 100) #non ci sono missing 

# Percentuale di NA per ogni variabile in test_df
sapply(test_df, function(x) sum(is.na(x)) / length(x) * 100) #non ci sono missing 

#### STANDARDIZZAZIONE VARIABILI CONTINUE #### 


numeric_cols <- names(train_df)[sapply(train_df, is.numeric)] 
# Standardizzazione delle variabili numeriche
train_df[numeric_cols] <- scale(train_df[numeric_cols])
test_df[numeric_cols] <- scale(test_df[numeric_cols])

# Controlliamo che la media sia circa 0 e la deviazione standard sia circa 1
summary(train_df[numeric_cols])
summary(test_df[numeric_cols])



#### ADDESTRAMENTO MODELLO NAIVE BAYES ####

# Modello con distribuzione normale
nb_model_norm <- NaiveBayes(cardio ~ ., data = train_df, usekernel = FALSE, laplace = 10)

# Modello con kernel density estimation
nb_model_kernel <- NaiveBayes(cardio ~ ., data = train_df, usekernel = TRUE, laplace = 10)


#### VALUTAZIONE MODELLO #### 

install.packages("pROC")
library(pROC)

# Previsioni su Test Set
predictions_norm <- predict(nb_model_norm, test_df, type = "class")
predictions_kernel <- predict(nb_model_kernel, test_df, type = "class")

# Matrice di confusione per entrambi i modelli
conf_matrix_norm <- confusionMatrix(predictions_norm$class, test_df$cardio)
conf_matrix_kernel <- confusionMatrix(predictions_kernel$class, test_df$cardio)

# Stampiamo le performance
print(conf_matrix_norm)
print(conf_matrix_kernel)


#### CURVE ROC PER ENTRAMBI I MODELLI #### 

# Predizioni per entrambi i modelli
pred_norm <- predict(nb_model_norm, test_df, type = "raw")$posterior[, 2]  # Probabilità per la classe "1"
pred_kernel <- predict(nb_model_kernel, test_df, type = "raw")$posterior[, 2]  # Probabilità per la classe "1"

# Valori reali della variabile target
true_labels <- as.numeric(as.character(test_df$cardio))  # Converto in numerico (0,1)

# ROC per il modello con distribuzione Normale
roc_norm <- roc(true_labels, pred_norm)

# ROC per il modello con Kernel Density Estimation
roc_kernel <- roc(true_labels, pred_kernel)

# Stampiamo l'AUC per entrambi i modelli
print(paste("AUC (Modello Normale):", round(auc(roc_norm), 3))) #auc = 0.771 
print(paste("AUC (Modello Kernel):", round(auc(roc_kernel), 3))) #auc = 0.768 

# Creazione del dataframe per ggplot
roc_data <- data.frame(
  FPR_Norm = 1 - roc_norm$specificities,
  TPR_Norm = roc_norm$sensitivities,
  FPR_Kernel = 1 - roc_kernel$specificities,
  TPR_Kernel = roc_kernel$sensitivities
)

# Grafico con ggplot2
ggplot() +
  geom_line(data = roc_data, aes(x = FPR_Norm, y = TPR_Norm), color = "blue", size = 1, linetype = "solid") +
  geom_line(data = roc_data, aes(x = FPR_Kernel, y = TPR_Kernel), color = "red", size = 1, linetype = "dashed") +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "gray") +
  labs(
    title = "Curva ROC - Confronto Modelli Naïve Bayes",
    x = "Falso Positivo Rate (1 - Specificità)",
    y = "Vero Positivo Rate (Sensibilità)"
  ) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC Normale:", round(auc(roc_norm), 3)), color = "blue") +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC Kernel:", round(auc(roc_kernel), 3)), color = "red")


#### Precision, Recall, F1-score #### 
precision <- conf_matrix_norm$byClass["Precision"]
recall <- conf_matrix_norm$byClass["Recall"] 
f1_score <- 2 * ((precision * recall) / (precision + recall))

print(paste("Precision:", round(precision, 2)))
print(paste("Recall:", round(recall, 2)))
print(paste("F1-score:", round(f1_score, 2))) 

#### ANALILSI DELL'IMPORTANZA BASATA SU PROB CONDIZIONATE ####

# Estrarre le tabelle di probabilità condizionate dal modello
prob_tables <- nb_model_norm$tables

# Funzione per calcolare la massima differenza tra le probabilità condizionate
calc_importance_prob <- function(prob_table) {
  if (is.matrix(prob_table)) {
    # Per variabili categoriche, calcoliamo la differenza massima tra le classi
    return(max(abs(prob_table[, 1] - prob_table[, 2])))
  } else {
    # Se è una variabile continua, restituiamo la differenza delle medie
    return(abs(prob_table[1,1] - prob_table[1,2]))  
  }
}

# Calcolare l'importanza di ogni variabile
importance_prob <- sapply(prob_tables, calc_importance_prob)

# Ordinare le variabili per importanza (differenza più alta tra le classi)
importance_sorted <- sort(importance_prob, decreasing = TRUE)

# Stampare le variabili più importanti
print(importance_sorted)




#### CONCLUSIONI ####

#> print(importance_sorted)
#age_years               ap_lo                 bmi               ap_hi              weight              height                alco 
#1.1119154           1.0887068           1.0808659           1.0663943           1.0528990           1.0193754           0.9102649 
#smoke                gluc         cholesterol              active         bp_category bp_category_encoded              gender 
#0.8470199           0.8228842           0.7318486           0.6364143           0.5758909           0.5758909           0.3096882 

#Le variabili più importanti sono fisiologiche:
  
#Età, pressione sanguigna (diastolica e sistolica), BMI e peso hanno il maggiore impatto sulla classificazione.
#Ha senso clinicamente, perché questi fattori sono direttamente collegati al rischio cardiovascolare.
#Il colesterolo è meno influente del previsto:
  
#Questo potrebbe dipendere dalla distribuzione dei dati o dal fatto che colesterolo è registrato in poche categorie (1, 2, 3), riducendone la variabilità.
#Variabili come "gender" hanno impatto basso:
#Probabilmente perché il dataset non mostra una differenza netta nel rischio cardiovascolare tra uomini e donne.









