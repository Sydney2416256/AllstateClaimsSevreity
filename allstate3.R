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



traindata$loss <- (traindata$loss+1)^.25

allstate_recipe <- recipe(loss ~ ., traindata) %>% 
  step_lencode_mixed(all_nominal_predictors(), 
                     outcome = vars(loss))

#create model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune(),
                          mode = "regression") %>%
  set_engine("lightgbm") 

#set workflow
boost_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(boost_model) 

#set up tuning grid
boost_tuneGrid <- grid_regular(tree_depth(), 
                               trees(), 
                               learn_rate(), 
                               levels = 5)

#set up cv
boost_folds <- vfold_cv(traindata, 
                        v = 10, 
                        repeats = 1)

CV_boost_results <- boost_wf %>%
  tune_grid(resamples = boost_folds,
            grid = boost_tuneGrid,
            metrics = metric_set(rmse))

#find best tuning parameters
bestTune_boost <- CV_boost_results %>%
  select_best(metric = "rmse") 

#finalize workflow and fit it
final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(train)

#make predictions
pred_boost <- predict(final_boost_wf, new_data = testdata)

#format for Kaggle
boost_final <- pred_boost %>%
  bind_cols(test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4-1)

write_csv(boost_final, "4Submission.csv")





