rm(list=ls())
options(scipen=999)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(ROCR)
library(pROC)

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


#### decision tree ####

#cambiamento target in fattoriale non numerico
train$cardio=ifelse(train$cardio==1,"yes","no")
test$cardio=ifelse(test$cardio==1,"yes","no")
train$cardio <- as.factor(train$cardio)
test$cardio <- as.factor(test$cardio)

# Costruzione dell'albero decisionale
set.seed(1) 
decision_tree <- rpart(cardio ~ ., data = train, method = "class", cp = 0.01)
rpart.plot(decision_tree, type = 4, extra = 1, main = "Decision Tree")

# Valutazione dell'importanza delle variabili
importance <- as.data.frame(varImp(decision_tree))
print(importance)

# Predizioni sul set di test
tree_predictions <- predict(decision_tree, test, type = "class")
confusionMatrix(tree_predictions, test$cardio, positive = "yes")

# Curva ROC e AUC
tree_probabilities <- predict(decision_tree, test, type = "prob")[, 2]  # Probabilità della classe positiva ("yes")
roc_curve <- roc(test$cardio, tree_probabilities, levels = c("no", "yes"))
plot(roc_curve, col = "blue", main = "ROC Curve - Decision Tree")
auc(roc_curve)

# Analisi della soglia ottimale
optimal_coords <- coords(roc_curve, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", optimal_coords$threshold, "\n")

# Ricalcolo della matrice di confusione con la soglia ottimale
optimal_tree_predictions <- ifelse(tree_probabilities > optimal_coords$threshold, "yes", "no")
optimal_tree_predictions <- as.factor(optimal_tree_predictions)
confusionMatrix(optimal_tree_predictions, test$cardio, positive = "yes")

#### decision tree - tuned ####

#valuazione dell'albero basata su accuracy
Ctrl <- trainControl(method = "cv" , number=10, classProbs = FALSE) 
set.seed(1)
tuned_model <- train(cardio ~ ., data = train, method = "rpart", 
                     tuneLength = 10, trControl = Ctrl, minsplit = 5)

custom_summary <- function(data, lev = NULL, model = NULL) 
{ sensitivity <- sensitivity(data$pred, data$obs, positive = "yes")
specificity <- specificity(data$pred, data$obs, positive = "yes")
accuracy <- sum(data$pred == data$obs) / length(data$obs)
out <- c(Sensitivity = sensitivity, Specificity = specificity, Accuracy = accuracy)
return(out)} 

#valutazione dell'albero basata su specificity
#       predict
#true   r0     r1
#r0     a      b
#r1     c      d   spec=d/(c+d)  
ctrl <- trainControl(method = "cv" , number=10, summaryFunction = custom_summary , classProbs = TRUE)
set.seed(1)
tuned_model_spec <- train(cardio ~ ., data = train, method = "rpart", 
                          tuneLength = 10, trControl=ctrl, minsplit = 5, metric="Specificity")

#valutazione dell'albero basata su sensitivity
ctrl <- trainControl(method = "cv" , number=10, summaryFunction = custom_summary , classProbs = TRUE)
set.seed(1)
tuned_model_sens <- train(cardio ~ ., data = train, method = "rpart", 
                          tuneLength = 10, trControl=ctrl, minsplit = 5, metric="Sensitivity")

#best tuned caret model
tuned_model
tuned_model_spec
tuned_model_sens
getTrainPerf(tuned_model)   #0.0014900662
getTrainPerf(tuned_model_spec)   #0.0014900662
getTrainPerf(tuned_model_sens)   #0.0014900662

#è sempre lo stesso albero (anche uguale a pruned.1)
tuned_decision_tree <- rpart(cardio ~ ., data = train, method = "class", cp = 0.0014900662)
rpart.plot(tuned_decision_tree, type = 4, extra = 1, main = "Decision Tree - Caret") 

#importance
importance3 <- as.data.frame(varImp(tuned_decision_tree))
print(importance3)
Vimportance3 <- varImp(tuned_model_sens)
plot(Vimportance3)

#validation caret
testpred <- predict(tuned_model_sens, test,type = "raw")
confusionMatrix(testpred, as.factor(test$cardio), positive="yes")

# Curva ROC e AUC
tuned_tree_probabilities <- predict(tuned_decision_tree, test, type = "prob")[, 2]  # Probabilità della classe positiva ("yes")
tuned_roc_curve <- roc(test$cardio, tuned_tree_probabilities, levels = c("no", "yes"))
plot(tuned_roc_curve, col = "blue", main = "ROC Curve - Decision Tree Caret")
auc(tuned_roc_curve)

# Analisi della soglia ottimale
tuned_optimal_coords <- coords(tuned_roc_curve, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", tuned_optimal_coords$threshold, "\n")

# Ricalcolo della matrice di confusione con la soglia ottimale
tuned_optimal_tree_predictions <- ifelse(tuned_tree_probabilities > tuned_optimal_coords$threshold, "yes", "no")
tuned_optimal_tree_predictions <- as.factor(tuned_optimal_tree_predictions)
confusionMatrix(tuned_optimal_tree_predictions, test$cardio, positive = "yes")



#### random forest ####

# Conversione delle variabili target e predittive in fattori (se necessario)
train$cardio <- as.factor(train$cardio)
test$cardio <- as.factor(test$cardio)

# Addestramento del modello Random Forest
set.seed(1)  
rf_model <- randomForest(cardio ~ ., data = train, ntree = 500, mtry = sqrt(ncol(train) - 1), importance = TRUE)
print(rf_model)

# Importanza delle variabili
varImpPlot(rf_model, main = "Variable Importance")

# Valutazione delle performance sul set di test
rf_predictions <- predict(rf_model, test, type = "class")
confusionMatrix(rf_predictions, test$cardio, positive = "yes")

# Curva ROC e AUC
rf_probabilities <- predict(rf_model, test, type = "prob")[, 2]  # Probabilità della classe positiva ("yes")
roc_curve <- roc(test$cardio, rf_probabilities, levels = c("no", "yes"))
plot(roc_curve, col = "blue", main = "ROC Curve - Random Forest")
auc(roc_curve)

# Analisi della soglia ottimale
optimal_coords <- coords(roc_curve, "best", ret = "all", transpose = FALSE)
cat("Optimal Threshold: ", optimal_coords$threshold, "\n")

# Ricalcolo della matrice di confusione con la soglia ottimale
optimal_rf_predictions <- ifelse(rf_probabilities > optimal_coords$threshold, "yes", "no")
optimal_rf_predictions <- as.factor(optimal_rf_predictions)
confusionMatrix(optimal_rf_predictions, test$cardio, positive = "yes")
