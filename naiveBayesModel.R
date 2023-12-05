library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(yardstick)



toView = data %>% head(100)

data = vroom("data.csv")
data2 = data %>% mutate(
  
  shot_made_flag = as.factor(shot_made_flag),
  season_team = str_c(season, opponent, sep = "_")
  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
               season_team, seconds_remaining,playoffs))



train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

bayesRegModel = naive_Bayes(Laplace = tune(), smoothness= tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
   step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
   # step_other(all_nominal_predictors(), threshold = .001) %>% 
   step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag)) 
  


# prepped = prep(my_recipe)
# 
# x = bake(prepped, new_data = train)

bayesReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(bayesRegModel)

tuning_grid = grid_regular(Laplace(), smoothness(), levels = 2)

folds = vfold_cv(train, v = 5, repeats = 1)

CV_results = bayesReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                             metrics = metric_set(mn_log_loss))

bestTune = CV_results %>% select_best("mn_log_loss")

final_wf = bayesReg_workflow %>% finalize_workflow(bestTune) %>% fit(train)


shot_preds = predict(final_wf, new_data = test, type = "prob")

sub = test %>% mutate(
  shot_made_flag = shot_preds$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)


#vroom_write(sub, "kobebayes2.csv", delim = ",")



