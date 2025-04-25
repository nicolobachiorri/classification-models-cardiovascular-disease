rm(list=ls())
setwd("/Users/nicolobachiorri/Desktop/PROGETTO ML ")
rm(list=ls())
library(mctest)
library(lmtest)
library(car) 
library(caret)
#install.packages("vcd")  # Se non è installato
library(vcd)
library(nnet)
library(caret)
library(dplyr)
library(NeuralNetTools)
library(ROCR)
library(pROC)
library(ipred)
set.seed(1) 

#### Preparazione dataset ####

r.tot <- read.csv("cardio_data_processed.csv", sep=",", header = T, stringsAsFactors = T, na.strings=c("NA","NaN", "","Unknown"))

# riclassificazione variabili ed eliminazione variabili inutili
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
r<-r[,-c(1,2)]  


#### Preprocessing reti neurali ####

#near zero variance
apply(r.tot, 2, function(x) length(unique(x)) == 1)
#missing data
sapply(r.tot, function(x)(sum(is.na(x))))
#model selection e collinearity (da albero)
r_ms <- r[,c("ap_hi","ap_lo", "bp_category", "cholesterol", "age_years", "cardio")]
#input transformation
r_ms$bp_category <- factor(r_ms$bp_category, 
                           levels = c("Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"), 
                           labels = c(0, 1, 2, 3))
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}
r_ms$ap_hi <- normalize(r_ms$ap_hi)
r_ms$ap_lo <- normalize(r_ms$ap_lo)
r_ms$age_years <- normalize(r_ms$age_years)
r_ms$bp_category <- normalize(as.numeric(r_ms$bp_category))
correlatedPredictors = findCorrelation(cor(r_ms[,c(1,2,5)]), cutoff = 0.95, names=TRUE)
correlatedPredictors
# zero variance
nzv = nearZeroVar(r_ms, saveMetrics = TRUE)
head(nzv[order(nzv$percentUnique, decreasing = FALSE), ], n = 20)

str(r_ms)

#### Divisione in Dataset di Training e Validation ####

set.seed(1)  
train.index.r <- sample(c(1:dim(r)[1]), dim(r)[1]*0.6)  
train <- r_ms[train.index.r, ]
test <- r_ms[-train.index.r, ]
train_ms<- r_ms[train.index.r, ]
test_ms <- r_ms[-train.index.r, ]

# cambiamento target in fattoriale non numerico
train$cardio=ifelse(train$cardio==1,"yes","no")
test$cardio=ifelse(test$cardio==1,"yes","no")
train$cardio <- as.factor(train$cardio)
test$cardio <- as.factor(test$cardio)

#### Percettrone ####

perceptron <-nnet(train_ms[,1:5],train_ms[,6],size=1,entropy=T)
perceptron
summary(perceptron)
plotnet(perceptron, alpha=0.6)
perceptron_predictions <- predict(perceptron, test_ms, type = "class")


# Modifica della classe positiva per le metriche di valutazione
conf_matrix <- confusionMatrix(as.factor(perceptron_predictions), as.factor(test_ms$cardio), positive = "1")
print(conf_matrix)

#### RETE NEURALE ####

neural_network <- nnet(train_ms[,1:5], train_ms[,6], entropy=T, size=3, maxit=2000, trace=T)
summary(neural_network)

plotnet(neural_network, alpha=0.6)

# Predizioni e correzione della classe positiva
cardiohat <- as.numeric(predict(neural_network, type='class'))
head(cardiohat)

# Assicurarsi che la classe positiva sia cardio = 1
table(cardiohat, train_ms$cardio)

# Importanza delle variabili
importance <- as.data.frame(varImp(neural_network))
print(importance)

# Validation
nn_predictions <- as.integer(predict(neural_network, test_ms, type = "class"))
conf_matrix <- table(nn_predictions, test_ms$cardio)

# Modifica della classe positiva per le metriche di valutazione
conf_matrix <- confusionMatrix(as.factor(nn_predictions), as.factor(test_ms$cardio), positive = "1")
print(conf_matrix)

# Curve ROC e AUC
nn_probabilities <- predict(neural_network, test_ms, type = "raw")
roc_nn <- roc(test_ms$cardio, nn_probabilities[, 1], levels = c("0", "1"), direction = "auto")
plot(roc_nn, col = "blue", main = "ROC Curve - Neural Network")
lines(roc_nn, col = "lightpink", lwd = 2)
auc(roc_nn)


# Analisi della soglia ottimale

# Ricalcolo della matrice di confusione con la soglia ottimale


#### Neural network con caret ####

# Metrica: Accuracy

ctrl <- trainControl(method="cv", number=5, search = "grid")
tuned_model <- train(train_ms[-6], as.factor(train$cardio),
                     method = "nnet",
                     preProcess = "range", 
                     metric="Accuracy", trControl=ctrl,
                     trace = TRUE, # use true to see convergence
                     maxit = 300)

# Metrica: specificity

custom_summary <- function(data, lev = NULL, model = NULL) 
{ sensitivity <- sensitivity(data$pred, data$obs)
specificity <- specificity(data$pred, data$obs)
accuracy <- sum(data$pred == data$obs) / length(data$obs)
out <- c(Sensitivity = sensitivity, Specificity = specificity, Accuracy = accuracy)
return(out)}
#train$cardio=ifelse(train$cardio==1,"yes","no")
#test$cardio=ifelse(test$cardio==1,"yes","no")
#train$cardio <- as.factor(train$cardio)
#test$cardio <- as.factor(test$cardio)

ctrl <- trainControl(method = "cv" , number=10, summaryFunction = custom_summary , classProbs = TRUE)
tuned_model_spec <- train(train[-6], as.factor(train$cardio),
                     method = "nnet",
                     preProcess = "range", 
                     metric="Specificity", trControl=ctrl,
                     trace = TRUE, # use true to see convergence
                     maxit = 300)
#?????
# Metrica: sensitivity

ctrl <- trainControl(method = "cv" , number=10, summaryFunction = custom_summary , classProbs = TRUE)
tuned_model_sens <- train(train[-6], as.factor(train$cardio),
                          method = "nnet",
                          preProcess = "range", 
                          metric="Sensitivity", trControl=ctrl,
                          trace = TRUE, # use true to see convergence
                          maxit = 300)

print(tuned_model)
plot(tuned_model)
getTrainPerf(tuned_model) 
print(tuned_model_spec)
plot(tuned_model_spec)
getTrainPerf(tuned_model_spec) 
print(tuned_model_sens)
plot(tuned_model_sens)
getTrainPerf(tuned_model_sens) 

#### RETE NEURALE TUNATA ####

neural_network <-nnet(train_ms[,1:5],train_ms[,6],size=5,decay=0,maxit=2000,trace=T,entropy=T)
neural_network
summary(neural_network)
plotnet(neural_network, alpha=0.6)
cardiohat <- as.numeric(predict(neural_network,type='class'))
head(cardiohat)
table(cardiohat, train$cardio)
confusionMatrix(as.factor(cardiohat), train_ms$cardio)

importance3 <- as.data.frame(varImp(neural_network))
print(importance3)

#validation
testpred <- predict(neural_network, test_ms,"class")
table(testpred, as.factor(test$cardio))

# curve roc e auc
head(predict(tuned_model, test, "prob"))
test$pred = predict(tuned_model, test, "prob")[,1]
length=roc(cardio ~ pred, data = test)
length
plot(length)


tuned_optimal_coords <- coords(length, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", tuned_optimal_coords$threshold, "\n")

#### PLS NEW ####

# Invertire i livelli del fattore cardio
train_ms$cardio <- relevel(as.factor(train_ms$cardio), ref = "1")
test_ms$cardio <- relevel(as.factor(test_ms$cardio), ref = "1")

# Train PLS con i nuovi livelli
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

pls <- train(
  cardio ~ ., data = train_ms,
  method = "pls",
  tuneGrid = expand.grid(ncomp = 5),
  trControl = ctrl, metric = "ROC"
)

# Predizioni
pls_predictions <- predict(pls, test_ms, type = "raw")

# Matrice di confusione
confusionMatrix(pls_predictions, test_ms$cardio)

# Curve ROC e AUC
tuned_pls_probabilities <- predict(pls, test_ms, type = "prob")[, 2]  # Ora la classe positiva è "1"
roc_pls <- roc(test_ms$cardio, tuned_pls_probabilities, levels = c("0", "1"), direction = "<")
plot(roc_pls, col = "blue", main = "ROC Curve - PLS Caret")
lines(roc_pls, col = "lightpink", lwd = 2)
auc(roc_pls)










#### PLS ####

train_ms$cardio <- as.factor(train_ms$cardio)
ctrl <- trainControl(method = "cv", number = 10)
pls <- train(
  cardio ~., data=train_ms,
  method = "pls",
  tuneGrid = expand.grid(ncomp = 5),  # Imposta ncomp a 5
  trControl = ctrl, metric="Accuracy")

# Validation

pls_predictions <- predict(pls, test_ms, type = "raw")
confusionMatrix(pls_predictions, as.factor(test_ms$cardio))

# Curve ROC e AUC
tuned_pls_probabilities <- predict(pls, test_ms, type = "prob")[, 2]  # Probabilità della classe positiva ("yes")
roc_pls <- roc(test_ms$cardio, tuned_pls_probabilities)
plot(roc_pls, col = "blue", main = "ROC Curve - PLS Caret")
lines(roc_pls, col = "lightpink", lwd = 2)
auc(roc_pls)

# Analisi della soglia ottimale
tuned_optimal_coords <- coords(roc_pls, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", tuned_optimal_coords$threshold, "\n")


# Ricalcolo della matrice di confusione con la soglia ottimale
tuned_optimal_pls_predictions <- ifelse(tuned_pls_probabilities > tuned_optimal_coords$threshold, "yes", "no")
tuned_optimal_pls_predictions <- as.factor(tuned_optimal_pls_predictions)
confusionMatrix(tuned_optimal_pls_predictions, test$cardio)







#### KNN ####



#preprocessing
r.tot <- read.csv("cardio_data_processed.csv", sep=",", header = T, stringsAsFactors = T, na.strings=c("NA","NaN", "","Unknown"))


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
r<-r[,-c(1,2)]  

#near zero variance
apply(r.tot, 2, function(x) length(unique(x)) == 1)
#missing data
sapply(r.tot, function(x)(sum(is.na(x))))
#model selection e collinearity (da albero)
r_ms <- r[,c("ap_hi","ap_lo", "bp_category", "cholesterol", "age_years", "cardio")]
#input transformation
r_ms$bp_category <- factor(r_ms$bp_category, 
                           levels = c("Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"), 
                           labels = c(0, 1, 2, 3))
correlatedPredictors = findCorrelation(cor(r_ms[,c(1,2,5)]), cutoff = 0.95, names=TRUE)
correlatedPredictors
# zero variance
nzv = nearZeroVar(r_ms, saveMetrics = TRUE)
head(nzv[order(nzv$percentUnique, decreasing = FALSE), ], n = 20)

str(r_ms)

#### Divisione in Dataset di Training e Validation ####

set.seed(1)  
train.index.r <- sample(c(1:dim(r)[1]), dim(r)[1]*0.6)  
train_ms <- r_ms[train.index.r, ]
test_ms <- r_ms[-train.index.r, ]
table(train$cardio)/nrow(train)
table(test$cardio)/nrow(test)





# tuning modello knn
cvCtrl <- trainControl(method = "cv", number=10, searc="grid", 
                       summaryFunction = twoClassSummary, 
                       classProbs = TRUE)
train_ms$cardio=ifelse(train_ms$cardio==1,"yes","no")
test_ms$cardio=ifelse(test_ms$cardio==1,"yes","no")
custom_summary <- function(data, lev = NULL, model = NULL) 
{ sensitivity <- sensitivity(data$pred, data$obs)
specificity <- specificity(data$pred, data$obs)
accuracy <- sum(data$pred == data$obs) / length(data$obs)
out <- c(Sensitivity = sensitivity, Specificity = specificity, Accuracy = accuracy)
return(out)}
knn <- train(cardio ~., data=train_ms,
                 method = "knn", tuneLength = 10,
                 preProcess = c("center", "scale", "corr"),
                 metric="Sensitivity",
                 trControl = cvCtrl)


# Validation

knn_predictions <- predict(knn, test_ms, type = "raw")
confusionMatrix(knn_predictions, as.factor(test_ms$cardio), positive = "yes") 


# Curve ROC e AUC
tuned_knn_probabilities <- predict(knn, test_ms, type = "prob")[, 2]  # Probabilità della classe positiva ("yes")
roc_knn <- roc(test_ms$cardio, tuned_knn_probabilities, levels = c("no", "yes"))
plot(roc_knn, col = "blue", main = "ROC Curve - KNN")
lines(roc_knn, col = "lightcoral", lwd = 2)
auc(roc_knn) 



# Analisi della soglia ottimale
tuned_optimal_coords <- coords(roc_knn, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", tuned_optimal_coords$threshold, "\n")


# Ricalcolo della matrice di confusione con la soglia ottimale
tuned_optimal_knn_predictions <- ifelse(tuned_knn_probabilities > tuned_optimal_coords$threshold, "yes", "no")
tuned_optimal_knn_predictions <- as.factor(tuned_optimal_knn_predictions)
confusionMatrix(tuned_optimal_knn_predictions, test$cardio, positive = "yes")


