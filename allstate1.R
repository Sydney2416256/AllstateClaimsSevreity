library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)
library(tidymodels)
library(dplyr)
library(poissonreg)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(embed)
library(discrim)
library(naivebayes)
library(bonsai)
library(lightgbm)
library(themis)
library(forecast)
library(corrplot)
library(recipes)


# Load in Data
data <- vroom("train.csv") |> 
  mutate(loss = log(loss))
testdata <- vroom("test.csv") 
colnames(data)



data |> 
  group_by(cat27) |> 
  summarise(cat_n = n())


ggplot(data, mapping = aes(x = loss,
                           y = cat27)) +
  geom_boxplot()

# Good: 3, 8-13, 19, 21-23
# Best: 7, 14, 16-18, 20
# cont3 as time series?
# cont14?


ggplot(data, mapping = aes(x = cont14,
                           y = loss)) +
  geom_point()+
  geom_smooth(method = "lm")

plot(data |> select(cont1, cont2, cont3, cont4, cont5, loss))
plot(data |> select(cont6, cont7, cont8, cont9, cont10, loss))






# take 10 random rows and any row column with the same values gets deleted
my_data <- data[sample(nrow(data), size = 10, replace = FALSE), ]
my_data_cleaned <- data[, sapply(my_data, function(x) length(unique(x)) > 1)]







 |> |> |> 

