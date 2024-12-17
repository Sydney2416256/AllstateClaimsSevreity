#load libraries
library(embed)
library(themis)
library(vroom)
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(bonsai)
library(lightgbm)

traindata <- vroom("STAT348/allstate/train.csv")
testdata <- vroom("STAT348/allstate/test.csv")



traindata <- traindata %>%
  mutate(loss = (loss+1)^.25)

allstate_recipe <- recipe(loss ~ ., traindata) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))



rf_model <- rand_forest(trees = tune(), 
                        min_n = tune(), 
                        mode = "regression") %>%
  set_engine("randomForest")

# Set up workflow
rf_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(rf_model)

# Set up tuning grid
rf_tuneGrid <- grid_regular(trees(), 
                            min_n(), 
                            levels = 3)

# Set up cross-validation
rf_folds <- vfold_cv(traindata, 
                     v = 3, 
                     repeats = 1)

# Tune the Random Forest model
CV_rf_results <- rf_wf %>%
  tune_grid(resamples = rf_folds,
            grid = rf_tuneGrid,
            metrics = metric_set(rmse))

# Find best tuning parameters
bestTune_rf <- CV_rf_results %>%
  select_best(metric = "rmse")

# Finalize workflow and fit the model
final_rf_wf <- rf_wf %>%
  finalize_workflow(bestTune_rf) %>%
  fit(traindata)

# Make predictions
pred_rf <- predict(final_rf_wf, new_data = testdata)

# Format for Kaggle submission
rf_final <- pred_rf %>%
  bind_cols(testdata) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4 - 1)  # Undo the transformation

write_csv(boost_final, "5Submission.csv")




