
# Financial Forecasting

library(readxl)
library(dplyr)
library(zoo)         # For lag function
library(tidyverse)
library(neuralnet)
library(Metrics)
library(MLmetrics)


# Read data
data = read_excel("C:/Users/geert/Desktop/ML cw/ExchangeUSD (2).xlsx")
exchange_rate <- data[["USD/EUR"]]

# Split the data into training and testing sets
train_set <- exchange_rate[1:400]
test_set <- exchange_rate[401:500]

anyNA(train_set)
head(train_set)
anyNA(test_set)
head(test_set)

# Normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Denormalization function
denormalize <- function(x, min, max) {
  return (x * (max - min) + min)
}

# Function to calculate performance indices
stat_indices <- function(predictions, actuals) {
  # Calculate RMSE
  rmse <- RMSE(predictions, actuals)
  
  # Calculate MAE
  mae <- MAE(predictions, actuals)
  
  # Calculate MAPE
  mape <- function(p, a) {
    ifelse(a == 0, 0, abs((p - a) / a) * 100)
  }
  mape_values <- mape(predictions, actuals)
  mape_result <- mean(mape_values, na.rm = TRUE)
  
  # Calculate SMAPE
  smape <- function(a, f) {
    return (1/length(a) * sum(2*abs(f-a) / (abs(a)+abs(f))*100))
  }
  smape_result <- smape(predictions, actuals)
  
  # Return the calculated statistics
  return(list("RMSE" = rmse, "MAE" = mae, "MAPE" = mape_result, "SMAPE" = smape_result))
}

# Obtain min and max values from original data
min_output <- min(exchange_rate)
max_output <- max(exchange_rate)
min_output
max_output

# Create lagged versions of train_set
lagged_train_set_4 <- data.frame(
  t_4 = lag(train_set, 4),
  t_3 = lag(train_set, 3),
  t_2 = lag(train_set, 2),
  t_1 = lag(train_set, 1),
  t = train_set
)

# Subset the lagged training set
IO_matrix_4_train <- lagged_train_set_4[complete.cases(lagged_train_set_4),]
head(IO_matrix_4_train)

# Create lagged versions of test_set
lagged_test_set_4 <- data.frame(
  t_4 = lag(test_set, 4),
  t_3 = lag(test_set, 3),
  t_2 = lag(test_set, 2),
  t_1 = lag(test_set, 1),
  t = test_set
)

# Subset the lagged test set
IO_matrix_4_test <- lagged_test_set_4[complete.cases(lagged_test_set_4),]
head(IO_matrix_4_test)

# Normalize training and testing data
IO_matrix_4_train_norm <- normalize(IO_matrix_4_train)
IO_matrix_4_test_norm <- normalize(IO_matrix_4_test)
head(IO_matrix_4_train_norm)
head(IO_matrix_4_test_norm)

actual_values <- IO_matrix_4_test_norm$t
actual_values_denorm_4 <- denormalize(actual_values, min_output, max_output)

# store performance in a list
performance_one_hidden = list()
performance_two_hidden = list()


# MLP model 1
set.seed(12)
mlp_model1 <- neuralnet(t ~ t_4 + t_3 + t_2 + t_1, data = IO_matrix_4_train_norm , hidden = c(5, 3), act.fct ="logistic", linear.output = FALSE)
plot(mlp_model1)
predictions_1 <- predict(mlp_model1, IO_matrix_4_test_norm)
predicted_values_denorm_1 <- denormalize(predictions_1, min_output, max_output)
performance_1 <- stat_indices(predicted_values_denorm_1, actual_values_denorm_4)
print(performance_1)
performance_two_hidden[[paste("input size 4  hidden layer 2 consists of 5 and 3. activation function logistic. nonlinear. ",sep = "")]] <- performance_1

# MLP model 2
set.seed(14)
mlp_model2 <- neuralnet(t ~ t_4 + t_3 + t_2 + t_1, data = IO_matrix_4_train_norm , hidden = c(8, 16), act.fct ="tanh", linear.output = FALSE)
plot(mlp_model2)
predictions_2 <- predict(mlp_model2, IO_matrix_4_test_norm)
predicted_values_denorm_2 <- denormalize(predictions_2, min_output, max_output)
performance_2 <- stat_indices(predicted_values_denorm_2, actual_values_denorm_4)
print(performance_2)
performance_two_hidden[[paste("input size 4  hidden layer 2 consists of 8 and 16. activation function tanh. nonlinear. ",sep = "")]] <- performance_2

# MLP model 3
set.seed(16)
mlp_model3 <- neuralnet(t ~ t_4 + t_3 + t_2 + t_1, data = IO_matrix_4_train_norm , hidden =  5, act.fct ="logistic", linear.output = TRUE)
plot(mlp_model3)
predictions_3 <- predict(mlp_model3, IO_matrix_4_test_norm)
predicted_values_denorm_3 <- denormalize(predictions_3, min_output, max_output)
performance_3 <- stat_indices(predicted_values_denorm_3, actual_values_denorm_4)
print(performance_3)
performance_one_hidden[[paste("input size 4  hidden layer 1 consists of 5. activation function logistic. linear. ",sep = "")]] <- performance_3

# MLP model 4
set.seed(18)
mlp_model4 <- neuralnet(t ~ t_4 + t_3 + t_2 + t_1, data = IO_matrix_4_train_norm , hidden =  7, act.fct ="tanh", linear.output = FALSE)
plot(mlp_model4)
predictions_4 <- predict(mlp_model4, IO_matrix_4_test_norm)
predicted_values_denorm_4 <- denormalize(predictions_4, min_output, max_output)
performance_4 <- stat_indices(predicted_values_denorm_4, actual_values_denorm_4)
print(performance_4)
performance_one_hidden[[paste("input size 4  hidden layer 1 consists of 7. activation function tanh. nonlinear. ",sep = "")]] <- performance_4




# Create lagged versions of train_set
lagged_train_set_3 <- data.frame(
  t_3 = lag(train_set, 3),
  t_2 = lag(train_set, 2),
  t_1 = lag(train_set, 1),
  t = train_set
)

# Subset the lagged training set
IO_matrix_3_train <- lagged_train_set_3[complete.cases(lagged_train_set_3),]

# Create lagged versions of test_set
lagged_test_set_3 <- data.frame(
  t_3 = lag(test_set, 3),
  t_2 = lag(test_set, 2),
  t_1 = lag(test_set, 1),
  t = test_set
)

# Subset the lagged test set
IO_matrix_3_test <- lagged_test_set_3[complete.cases(lagged_test_set_3),]

# Normalize training and testing data
IO_matrix_3_train_norm <- normalize(IO_matrix_3_train)
IO_matrix_3_test_norm <- normalize(IO_matrix_3_test)

actual_values_3 <- IO_matrix_3_test_norm$t
actual_values_denorm_3 <- denormalize(actual_values_3, min_output, max_output)

# MLP model 5
set.seed(12)
mlp_model5 <- neuralnet(t ~ t_3 + t_2 + t_1, data = IO_matrix_3_train_norm , hidden = c(5, 8), act.fct ="logistic", linear.output = TRUE)
plot(mlp_model5)
predictions_5 <- predict(mlp_model5, IO_matrix_3_test_norm)
predicted_values_denorm_5 <- denormalize(predictions_5, min_output, max_output)
performance_5 <- stat_indices(predicted_values_denorm_5, actual_values_denorm_3)
print(performance_5)
performance_two_hidden[[paste("input size 3  hidden layer 2 consists of 5 and 8. activation function logistic. linear. ",sep = "")]] <- performance_5

# MLP model 6
set.seed(14)
mlp_model6 <- neuralnet(t ~ t_3 + t_2 + t_1, data = IO_matrix_3_train_norm , hidden = c(6, 4), act.fct ="tanh", linear.output = FALSE)
plot(mlp_model6)
predictions_6 <- predict(mlp_model6, IO_matrix_3_test_norm)
predicted_values_denorm_6 <- denormalize(predictions_6, min_output, max_output)
performance_6 <- stat_indices(predicted_values_denorm_6, actual_values_denorm_3)
print(performance_6)
performance_two_hidden[[paste("input size 3  hidden layer 2 consists of 6 and 4. activation function tanh. nonlinear. ",sep = "")]] <- performance_6

# MLP model 7
set.seed(16)
mlp_model7 <- neuralnet(t ~ t_3 + t_2 + t_1, data = IO_matrix_3_train_norm , hidden =  5, act.fct ="logistic", linear.output = FALSE)
plot(mlp_model7)
predictions_7 <- predict(mlp_model7, IO_matrix_3_test_norm)
predicted_values_denorm_7 <- denormalize(predictions_7, min_output, max_output)
performance_7 <- stat_indices(predicted_values_denorm_7, actual_values_denorm_3)
print(performance_7)
performance_one_hidden[[paste("input size 3  hidden layer 1 consists of 5. activation function logistic. nonlinear. ",sep = "")]] <- performance_7

# MLP model 8
set.seed(18)
mlp_model8 <- neuralnet(t ~ t_3 + t_2 + t_1, data = IO_matrix_3_train_norm , hidden =  6, act.fct ="tanh", linear.output = FALSE)
plot(mlp_model8)
predictions_8 <- predict(mlp_model8, IO_matrix_3_test_norm)
predicted_values_denorm_8 <- denormalize(predictions_8, min_output, max_output)
performance_8 <- stat_indices(predicted_values_denorm_8, actual_values_denorm_3)
print(performance_8)
performance_one_hidden[[paste("input size 3  hidden layer 1 consists of 6. activation function tanh. nonlinear. ",sep = "")]] <- performance_8



# Create lagged versions of train_set
lagged_train_set_2 <- data.frame(
  t_2 = lag(train_set, 2),
  t_1 = lag(train_set, 1),
  t = train_set
)

# Subset the lagged training set
IO_matrix_2_train <- lagged_train_set_2[complete.cases(lagged_train_set_2),]

# Create lagged versions of test_set
lagged_test_set_2 <- data.frame(
  t_2 = lag(test_set, 2),
  t_1 = lag(test_set, 1),
  t = test_set
)

# Subset the lagged test set
IO_matrix_2_test <- lagged_test_set_2[complete.cases(lagged_test_set_2),]

# Normalize training and testing data
IO_matrix_2_train_norm <- normalize(IO_matrix_2_train)
IO_matrix_2_test_norm <- normalize(IO_matrix_2_test)

actual_values_2 <- IO_matrix_2_test_norm$t
actual_values_denorm_2 <- denormalize(actual_values_2, min_output, max_output)

# MLP model 9
set.seed(12)
mlp_model9 <- neuralnet(t ~ t_2 + t_1, data = IO_matrix_2_train_norm , hidden = c(5, 2), act.fct ="logistic", linear.output = FALSE)
plot(mlp_model9)
predictions_9 <- predict(mlp_model9, IO_matrix_2_test_norm)
predicted_values_denorm_9 <- denormalize(predictions_9, min_output, max_output)
performance_9 <- stat_indices(predicted_values_denorm_9,actual_values_denorm_2)
print(performance_9)
performance_two_hidden[[paste("input size 2  hidden layer 2 consists of 5 and 2. activation function logistic. nonlinear. ",sep = "")]] <- performance_9

# MLP model 10
set.seed(14)
mlp_model10 <- neuralnet(t ~ t_2 + t_1, data = IO_matrix_2_train_norm , hidden = c(4, 6), act.fct ="tanh", linear.output = FALSE)
plot(mlp_model10)
predictions_10 <- predict(mlp_model10, IO_matrix_2_test_norm)
predicted_values_denorm_10 <- denormalize(predictions_10, min_output, max_output)
performance_10 <- stat_indices(predicted_values_denorm_10, actual_values_denorm_2)
print(performance_10)
performance_two_hidden[[paste("input size 2  hidden layer 2 consists of 4 and 6. activation function tanh. nonlinear. ",sep = "")]] <- performance_10

# MLP model 11
set.seed(16)
mlp_model11 <- neuralnet(t ~ t_2 + t_1, data = IO_matrix_2_train_norm , hidden =  5, act.fct ="logistic", linear.output = FALSE)
plot(mlp_model11)
predictions_11 <- predict(mlp_model11, IO_matrix_2_test_norm)
predicted_values_denorm_11 <- denormalize(predictions_11, min_output, max_output)
performance_11 <- stat_indices(predicted_values_denorm_11, actual_values_denorm_2)
print(performance_11)
performance_one_hidden[[paste("input size 2  hidden layer 1 consists of 5. activation function logistic. nonlinear. ",sep = "")]] <- performance_11

# MLP model 12
set.seed(18)
mlp_model12 <- neuralnet(t ~ t_2 + t_1, data = IO_matrix_2_train_norm , hidden =  8, act.fct ="tanh", linear.output = TRUE)
plot(mlp_model12)
predictions_12 <- predict(mlp_model12, IO_matrix_2_test_norm)
predicted_values_denorm_12 <- denormalize(predictions_12, min_output, max_output)
performance_12 <- stat_indices(predicted_values_denorm_12, actual_values_denorm_2)
print(performance_12)
performance_one_hidden[[paste("input size 2  hidden layer 1 consists of 8. activation function tanh. linear. ",sep = "")]] <- performance_12


# Create lagged versions of train_set
lagged_train_set_1 <- data.frame(
  t_1 = lag(train_set, 1),
  t = train_set
)

# Subset the lagged training set
IO_matrix_1_train <- lagged_train_set_1[complete.cases(lagged_train_set_1),]

# Create lagged versions of test_set
lagged_test_set_1 <- data.frame(
  t_1 = lag(test_set, 1),
  t = test_set
)

# Subset the lagged test set
IO_matrix_1_test <- lagged_test_set_1[complete.cases(lagged_test_set_1),]

# Normalize training and testing data
IO_matrix_1_train_norm <- normalize(IO_matrix_1_train)
IO_matrix_1_test_norm <- normalize(IO_matrix_1_test)

actual_values_1 <- IO_matrix_1_test_norm$t
actual_values_denorm_1 <- denormalize(actual_values_1, min_output, max_output)


# MLP model 13
set.seed(12)
mlp_model13 <- neuralnet(t ~ t_1, data = IO_matrix_1_train_norm , hidden = c(6, 8), act.fct ="logistic", linear.output = FALSE)
plot(mlp_model13)
predictions_13 <- predict(mlp_model13, IO_matrix_1_test_norm)
predicted_values_denorm_12 <- denormalize(predictions_13, min_output, max_output)
performance_13 <- stat_indices(predicted_values_denorm_12 , actual_values_denorm_1)
print(performance_13)
performance_two_hidden[[paste("input size 1  hidden layer 2 consists of 6 and 8. activation function logistic. nonlinear. ",sep = "")]] <- performance_13

# MLP model 14
set.seed(16)
mlp_model14 <- neuralnet(t ~ t_1, data = IO_matrix_1_train_norm , hidden =  5, act.fct ="tanh", linear.output = FALSE)
plot(mlp_model14)
predictions_14 <- predict(mlp_model14, IO_matrix_1_test_norm)
predicted_values_denorm_14 <- denormalize(predictions_14, min_output, max_output)
performance_14 <- stat_indices(predicted_values_denorm_14, actual_values_denorm_1)
print(performance_14)
performance_one_hidden[[paste("input size 1  hidden layer 1 consists of 5. activation function tanh. nonlinear. ",sep = "")]] <- performance_14

# Print results for one hidden layer
print(performance_one_hidden)

# Print results for two hidden layer
print(performance_two_hidden)

#################################################################################

#g

# Calculate total parameters


# Best one-hidden layer network
input_size_1 <- 4
hidden_layer_size_1 <- 5
output_size_1 <- 1

total_parameters_1 <- (input_size_1 + 1) * hidden_layer_size_1 + (hidden_layer_size_1 + 1) * output_size_1
total_parameters_1
cat("Total parameters for one hidden layer: ", total_parameters_1, "\n")

# Best two-hidden layer network
input_size_2 <- 2  
hidden_layer_size1_2 <- 4
hidden_layer_size_2 <- 6
output_size_2 <- 1

total_parameters_2 <- (input_size_2 + 1) * hidden_layer_size1_2 + (hidden_layer_size1_2 + 1) * hidden_layer_size_2 + (hidden_layer_size_2 + 1) * output_size_2
total_parameters_2
cat("Total parameters for two hidden layer: ", total_parameters_2, "\n")


#################################################################################

#h

# Initialize variables to store the best model 1 and its RMSE
best_model_1 = ""
best_rmse_1 = Inf
# Iterate over the stored results
for (key in names(performance_one_hidden)) {
  RMSE = performance_one_hidden[[key]]$RMSE
  
  # Check if the current model has a lower RMSE than the best model so far
  if (RMSE < best_rmse_1) {
    best_rmse_1 = RMSE
    best_model_1 = key
  }
}

# Print the best model and its RMSE
cat("Best model for one hidden layer:", best_model_1, "\n")
cat("Best RMSE:", best_rmse_1, "\n")


# Initialize variables to store the best model 2 and its RMSE
best_model_2 = ""
best_rmse_2 = Inf

# Iterate over the stored results
for (key in names(performance_two_hidden)) {
  RMSE = performance_two_hidden[[key]]$RMSE
  
  # Check if the current model has a lower RMSE than the best model so far
  if (RMSE < best_rmse_2) {
    best_rmse_2 = RMSE
    best_model_2 = key
  }
}

# Print the best model and its RMSE
cat("Best model for two hidden layer:", best_model_2 , "\n")
cat("Best RMSE:", best_rmse_2, "\n")

# Denormalized values
head(actual_values_denorm_4)
head(predicted_values_denorm_3)

# Plot the results
plot(actual_values_denorm_4, predicted_values_denorm_3,
     xlab = "Desired Output", ylab = "Predicted Output", 
     main = "Predicted vs Desired One hidden layer", col = "red", pch = 19)
abline(0, 1, col = "darkblue")
legend("bottomright", legend = "Expected Line", col = "darkblue", lty = 1, cex = 0.8)


# Line chart of predicted vs. expected values
plot(actual_values_denorm_4, type = "l", col = "darkblue", ylim = range(c(actual_values_denorm_4,
                                                         predicted_values_denorm_3)))
lines(predicted_values_denorm_3, col = "red")

