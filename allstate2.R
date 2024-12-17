
data <- vroom("STAT348/allstate/train.csv") |> 
  mutate(loss = log(loss))
testdata <- vroom("STAT348/allstate/test.csv") 

traindata <- data[sample(nrow(data), size = 100, replace = FALSE), ]

my_recipe <- recipe(loss ~ ., traindata) |> 
  step_rm(id) %>%  # Remove ID column as it's not predictive
  step_other(all_nominal_predictors(), threshold = .001) %>%  # Combine rare levels in categorical predictors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%  # Apply target encoding to categorical variables
  step_corr(all_numeric_predictors(), threshold = 0.6) %>%  # Remove highly correlated numeric variables
  step_normalize(all_numeric_predictors())  |> 
  step_zv(all_predictors()) %>%  # Remove zero-variance predictors first
  



















# Summary of the recipe
summary(my_recipe)


mod <- boost_tree(
  mode = "regression",
  engine = "xgboost",
  trees = 25,
  min_n = tune(),
  tree_depth = tune(),
)


wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(mod)

grid_of_tuning_params <- grid_regular(min_n(),
                                      tree_depth(),
                                      levels = 5) 

## Split data for CV
folds <- vfold_cv(traindata, v = 5, repeats=1)

## Run CV
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse))


## Get Best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)




pred <- predict(final_wf, new_data = testdata)
predictions <- exp(pred)


kaggle_submission <- predictions %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(id, .pred) %>% #Just keep datetime and prediction variables
  rename(loss=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(loss=pmax(0, loss)) %>% #pointwise max of (0, prediction)
  
## Write out the file
vroom_write(x=kaggle_submission, file="./Allstate_Bart.csv", delim=",")

