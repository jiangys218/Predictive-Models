library(readr)
library(corrplot)
library(tidyverse)
library(stringr)
library(lubridate)
library(DT)
library(caret)
library(leaflet)
library(boot)
library(dplyr)
library(glmnet)
library(GGally)
library(penalized)
library(randomForest)
library(xgboost)
library(plotmo)

################################# prepare data #############################################
kcdata = read_csv("Desktop/kc_house_data.csv")              #house price dataset
crimedata = read.csv("~/Desktop/King_County_Sheriff_s_Office_-_Incident_Dataset.csv")  #crime rate dataset
kcpop = read.csv("~/Desktop/King_County_Population.csv")    #population dataset

crimerate = crimedata %>% group_by(zip) %>% summarize(n())  #count number of crime per zipcode
kcdata$zipcode = as.factor(kcdata$zipcode)
colnames(crimerate) = c("zipcode", "countcrime")
kcdata = left_join(kcdata, crimerate, by="zipcode")         #join crimerate to kcdata
kcdata$countcrime.y[which(is.na(kcdata$countcrime.y))] = 0  #change NA to 0 crime 
kcdata = left_join(kcdata, kcpop, by="zipcode")             #join population to kcdata
kcdata = kcdata[,c(-1,-2)]                                  #remove id and date
sum(is.na(kcdata))                                          #checkforna


########################## clean data format ###################################

kcdata$sqft_above <- kcdata$sqft_above/kcdata$sqft_living
kcdata$yr_built <- 2020 - kcdata$yr_built
kcdata$yr_renovated <- !(kcdata$yr_renovated == 0)
kcdata$zipcode <- as.factor(kcdata$zipcode)


##################### map #############################################

kcdata$PriceBin<-cut(kcdata$price, c(0,250000,500000,750000,1000000,2000000,999000000))

center_lon = median(kcdata$long,na.rm = TRUE)
center_lat = median(kcdata$lat,na.rm = TRUE)

factpal <- colorFactor(c("black","blue","yellow","orange","#0B5345","red"), 
                       kcdata$PriceBin)

leaflet(kcdata) %>% addProviderTiles("Esri.NatGeoWorldMap") %>%
  addCircles(lng = ~long, lat = ~lat, 
             color = ~factpal(PriceBin))  %>%
  # controls
  setView(lng=center_lon, lat=center_lat,zoom = 12) %>%
  
  addLegend("bottomright", pal = factpal, values = ~PriceBin,
            title = "House Price Distribution",
            opacity = 1)


################################ feature selection ##########################################

############ 1. correlation
corr = cor(kcdata)
corrplot(corr, method = "color", outline = T, cl.pos = 'n', rect.col = "black",  
         tl.col = "indianred4", addCoef.col = "black", number.digits = 2, 
         number.cex = 0.60, tl.cex = 0.7, cl.cex = 1, 
         col = colorRampPalette(c("green4","white","red"))(100))


#highly correlation variables: 
#sqft_living -> bathroom(0.75) -> grade(0.76) -> sqft_above(0.88) -> sqft_living_15(0.76)
#sqft_lot -> sqft_lot_15(0.72)

ggpairs(kcdata, title="correlogram") 

############ 2. Recursive Feature Elimination
rfecontrol = rfeControl(functions = lmFuncs,
           method = "repeatedcv",
           repeats = 5,
           verbose = FALSE)
results <- rfe(kcdata, as.matrix(kcdata[,1]), sizes=c(1:21), rfeControl=rfecontrol)

#Recursive feature selection

#Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 

#Resampling performance over subset size:
  
#  Variables   RMSE Rsquared    MAE RMSESD RsquaredSD MAESD Selected
#1 353472  0.07315 229598  19619    0.03172  4557         
#2 334508  0.16839 207134  20085    0.02926  5079         
#3 333352  0.17407 204822  20149    0.02963  4979         
#4 245651  0.55234 154183  17945    0.02241  3782         
#5 238154  0.57939 147900  18324    0.02306  3936         
#6 233786  0.59466 146207  17645    0.02181  3844         
#7 232986  0.59747 145515  17672    0.02157  3810         
#8 229425  0.60979 143031  17876    0.02234  3848         
#9 224201  0.62742 139396  18342    0.02716  5448         
#10 218008  0.64781 134444  16884    0.02096  3288         
#11 216506  0.65262 134294  17005    0.02183  3416         
#12 201823  0.69818 126298  13897    0.01631  2924         
#13 200810  0.70123 125372  13936    0.01647  2928         
#14 200537  0.70203 125492  13903    0.01636  2925         
#15 200560  0.70198 125424  14015    0.01640  2949         
#16 200277  0.70283 125220  14062    0.01643  2952         
#17 199164  0.70616 124411  13934    0.01589  2884         
#18 199008  0.70662 124408  13865    0.01594  2859         
#19 198994  0.70666 124376  13861    0.01587  2835         
#20 198994  0.70666 124376  13861    0.01587  2835        *
  
#  The top 5 variables (out of 20):
#  waterfront, lat, long, grade, view

predictors(results)
#[1] "waterfront"    "lat"           "long"          "grade"         "view"         
#[6] "bathrooms"     "bedrooms"      "condition"     "floors"        "yr_built"     
#[11] "zipcode"       "sqft_living"   "countcrime.y"  "sqft_above"    "sqft_living15"
#[16] "yr_renovated"  "Population"    "sqft_lot15"    "sqft_lot"      "sqft_basement"
plot(results)

#based on the graph, 12 predictors should be optimal

############################## Building Models ##################################

fitControl <- trainControl(method="cv",number = 5) 

################################## LM ######################################################

###### Fit all predictors
lm.all <- lm(price~.,data=kcdata)


###### Use step function
step <- stepAIC(lm.all, direction="both")
step$anova


###### Fit best model suggest by stepAIC
lm.fit <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
  waterfront + view + condition + grade + sqft_above + yr_built + 
  yr_renovated + zipcode + lat + long + sqft_living15 + sqft_lot15, data=kcdata)
summary(lm.fit)
plot(lm.fit)

###### Fit best model suggest by stepAIC with log(price)
lm.fit.log <- lm(log(price) ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
               waterfront + view + condition + grade + sqft_above + yr_built + 
               yr_renovated + zipcode + lat + long + sqft_living15 + sqft_lot15, data=kcdata)
summary(lm.fit.log)
plot(lm.fit.log)

#### check for important predictors
lm.all = train(price ~ ., data = kcdata, method = "lm",trControl = fitControl, metric="RMSE")
imp.lm.all <- varImp(lm.all, scale=FALSE)
plot(imp.lm.all)



################################ Penalized (for interaction terms) ####################################

lm.pen.all = train(price ~ ., data = kcdat, method = "penalized",trControl = fitControl, metric="RMSE")
imp.lm.pen.all <- varImp(lm.pen.all, scale=FALSE)
plot(imp.lm.pen.all)

#Penalized Linear Regression 

#21613 samples
#20 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17291, 17290, 17292, 17290, 17289 
#Resampling results across tuning parameters:
  
#  lambda1  lambda2  RMSE      Rsquared   MAE     
#1        1        199466.7  0.7059935  124497.9
#1        2        199467.9  0.7059864  124484.2
#1        4        199474.0  0.7059617  124460.5
#2        1        199466.7  0.7059935  124497.9
#2        2        199467.9  0.7059864  124484.2
#2        4        199474.0  0.7059617  124460.5
#4        1        199466.7  0.7059935  124497.9
#4        2        199467.9  0.7059864  124484.2
#4        4        199474.0  0.7059617  124460.5

#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were lambda1 = 4 and lambda2 = 1.


################################# random forest regression ##########################

rf.all = train(price ~ ., data = kcdata, method = "rf",trControl = fitControl, metric="RMSE")
imp.rf.all <- varImp(rf.all, scale=FALSE)
plot(imp.rf.all)

#Random Forest 

#21613 samples
#20 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17291, 17290, 17291, 17291, 17289 
#Resampling results across tuning parameters:
  
#  mtry  RMSE      Rsquared   MAE     
#2    134867.2  0.8758343  71284.70
#11    126454.1  0.8836143  67465.99
#20    127146.4  0.8815740  68194.30

#RMSE was used to select the optimal model using the smallest value.
##The final value used for the model was mtry = 11.

rf.model = randomForest(price~., data=kcdata, mtry = 11)


################################# GLR #############################################

glmnet.all = train(price ~ ., data = kcdat,
                         method = "glmnet",trControl = fitControl,metric="RMSE")

imp.glmnet.all <- varImp(glmnet.all, scale=FALSE)
plot(imp.glmnet.all)


#21613 samples
#20 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17290, 17291, 17291, 17291, 17289 
#Resampling results across tuning parameters:
  
#  alpha  lambda      RMSE      Rsquared   MAE     
#0.10     515.4604  199311.0  0.7056091  124095.9
#0.10    5154.6040  199351.0  0.7055418  123688.5
#0.10   51546.0398  203141.2  0.6984898  121461.4
#0.55     515.4604  199315.2  0.7055893  124172.5
#0.55    5154.6040  199733.1  0.7047247  122851.2
#0.55   51546.0398  216271.1  0.6744996  126079.7
#1.00     515.4604  199325.2  0.7055698  124124.0
#1.00    5154.6040  200622.3  0.7025622  122520.3
#1.00   51546.0398  233183.8  0.6336587  138479.8

#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were alpha = 0.1 and lambda = 515.4604.

glmnet.fit.log = train(log(price) ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
                     waterfront + view + condition + grade + sqft_above + yr_built + 
                     yr_renovated + zipcode + lat + long + sqft_living15 + sqft_lot15, 
                   data = kcdata, method = "glmnet",trControl = fitControl,metric="RMSE")
#glmnet 

#21613 samples
#17 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17290, 17292, 17290, 17289, 17291 
#Resampling results across tuning parameters:
  
#  alpha  lambda        RMSE       Rsquared   MAE      
#0.10   0.0007411693  0.2526281  0.7698551  0.1959454
#0.10   0.0074116925  0.2527233  0.7697460  0.1960614
#0.10   0.0741169253  0.2586986  0.7638683  0.2012060
#0.55   0.0007411693  0.2526275  0.7698518  0.1959346
#0.55   0.0074116925  0.2533591  0.7688333  0.1964114
#0.55   0.0741169253  0.2784761  0.7406933  0.2171302
#1.00   0.0007411693  0.2526586  0.7697992  0.1959728
#1.00   0.0074116925  0.2543856  0.7673418  0.1970426
#1.00   0.0741169253  0.2992769  0.7153145  0.2338770

#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were alpha = 0.55 and lambda = 0.0007411693.

x <- model.matrix(price ~ ., kcdat)[, -1]
glmnet.model = glmnet(x, kcdata$price, alpha = 0.1)
plot(glmnet.model,xvar = "lambda", main = "Elastic Net (Alpha = 0.55)\n")

 
plot_glmnet(glmnet.model)

###################################### xgboost #########################################

xgb.log.all = train(log(price) ~ ., data = kcdata, method = "xgbTree",trControl = fitControl, metric="RMSE")
imp.xgb.all <- varImp(xgb.all, scale=FALSE)
plot(imp.xgb.all)

#eXtreme Gradient Boosting 

#21613 samples
#20 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17291, 17290, 17289, 17290, 17292 
#Resampling results across tuning parameters:
  
#  eta  max_depth  colsample_bytree  subsample  nrounds  RMSE      Rsquared   MAE      
#0.3  1          0.6               0.50        50      178352.3  0.7650333  103997.29
#0.3  1          0.6               0.50       100      172175.0  0.7810522   99659.67
#0.3  1          0.6               0.50       150      167350.1  0.7936044   96930.09
#0.3  1          0.6               0.75        50      177285.4  0.7689207  103488.74
#0.3  1          0.6               0.75       100      168381.0  0.7904157   98233.09
#0.3  1          0.6               0.75       150      164112.6  0.8013117   95623.61
#0.3  1          0.6               1.00        50      177426.9  0.7685273  103629.04
#0.3  1          0.6               1.00       100      167491.6  0.7927132   97846.30
#0.3  1          0.6               1.00       150      163686.9  0.8023173   95140.10
#0.3  1          0.8               0.50        50      177667.9  0.7671878  103396.27
#0.3  1          0.8               0.50       100      170986.8  0.7842259   99380.40
#0.3  1          0.8               0.50       150      167989.9  0.7920058   97572.68
#0.3  1          0.8               0.75        50      177166.0  0.7687192  102648.16
#0.3  1          0.8               0.75       100      168465.9  0.7907655   97910.01
#0.3  1          0.8               0.75       150      165084.5  0.7992721   95256.48
#0.3  1          0.8               1.00        50      174402.5  0.7769174  102267.80
#0.3  1          0.8               1.00       100      165048.8  0.7987324   96676.66
#0.3  1          0.8               1.00       150      161356.3  0.8078772   94225.64
#0.3  2          0.6               0.50        50      148845.4  0.8362656   88370.25
#0.3  2          0.6               0.50       100      138337.8  0.8581826   82248.15
#0.3  2          0.6               0.50       150      134132.8  0.8666822   79107.41
#0.3  2          0.6               0.75        50      144467.0  0.8460580   85885.71
#0.3  2          0.6               0.75       100      136477.0  0.8626814   80267.24
#0.3  2          0.6               0.75       150      131493.7  0.8725309   76729.94
#0.3  2          0.6               1.00        50      146190.3  0.8426190   85733.57
#0.3  2          0.6               1.00       100      137629.2  0.8607161   79482.03
#0.3  2          0.6               1.00       150      132953.2  0.8701452   76084.41
#0.3  2          0.8               0.50        50      146150.7  0.8418912   86945.02
#0.3  2          0.8               0.50       100      138514.9  0.8584551   81693.21
#0.3  2          0.8               0.50       150      134792.2  0.8660332   78935.43
#0.3  2          0.8               0.75        50      145702.4  0.8434478   85947.99
#0.3  2          0.8               0.75       100      138060.8  0.8598274   80327.38
#0.3  2          0.8               0.75       150      134158.7  0.8678665   77248.90
#0.3  2          0.8               1.00        50      144981.9  0.8456180   85539.22
#0.3  2          0.8               1.00       100      136091.0  0.8641038   79179.22
#0.3  2          0.8               1.00       150      131117.5  0.8738883   75629.83
#0.3  3          0.6               0.50        50      134833.1  0.8659027   78969.71
#0.3  3          0.6               0.50       100      128795.0  0.8776330   74916.98
#0.3  3          0.6               0.50       150      125607.1  0.8834650   72714.76
#0.3  3          0.6               0.75        50      133298.1  0.8689490   77705.49
#0.3  3          0.6               0.75       100      126954.5  0.8812135   73074.41
#0.3  3          0.6               0.75       150      124867.3  0.8850952   70962.81
#0.3  3          0.6               1.00        50      132580.8  0.8701895   77139.49
#0.3  3          0.6               1.00       100      125832.1  0.8833981   72225.40
#0.3  3          0.6               1.00       150      123364.7  0.8880311   70252.67
#0.3  3          0.8               0.50        50      136632.6  0.8619521   79740.04
#0.3  3          0.8               0.50       100      132816.1  0.8699119   75865.45
#0.3  3          0.8               0.50       150      130639.6  0.8742235   74195.19
#0.3  3          0.8               0.75        50      133065.1  0.8697852   77522.87
#0.3  3          0.8               0.75       100      128315.2  0.8793059   73121.04
#0.3  3          0.8               0.75       150      126191.4  0.8833782   71478.75
#0.3  3          0.8               1.00        50      131554.1  0.8725876   76730.97
#0.3  3          0.8               1.00       100      125243.3  0.8845709   72107.44
#0.3  3          0.8               1.00       150      122761.1  0.8892215   69899.73
#0.4  1          0.6               0.50        50      176214.2  0.7701448  104871.50
#0.4  1          0.6               0.50       100      170269.6  0.7857476   99738.81
#0.4  1          0.6               0.50       150      166244.7  0.7966573   97049.80
#0.4  1          0.6               0.75        50      176047.0  0.7703480  103749.22
#0.4  1          0.6               0.75       100      168177.0  0.7913574   97836.77
#0.4  1          0.6               0.75       150      163495.4  0.8030340   95085.13
#0.4  1          0.6               1.00        50      175762.6  0.7714530  103585.54
#0.4  1          0.6               1.00       100      166575.2  0.7952262   97509.41
#0.4  1          0.6               1.00       150      162684.6  0.8048737   94809.51
#0.4  1          0.8               0.50        50      179558.9  0.7613374  105032.82
#0.4  1          0.8               0.50       100      174027.4  0.7768736  100093.09
#0.4  1          0.8               0.50       150      168851.1  0.7903530   97198.34
#0.4  1          0.8               0.75        50      174992.3  0.7736298  103460.69
#0.4  1          0.8               0.75       100      169153.6  0.7897154   97722.15
#0.4  1          0.8               0.75       150      165013.2  0.8001845   95019.08
#0.4  1          0.8               1.00        50      175063.6  0.7734147  103763.59
#0.4  1          0.8               1.00       100      165757.3  0.7972630   97364.19
#0.4  1          0.8               1.00       150      162249.7  0.8059665   94776.56
#0.4  2          0.6               0.50        50      145835.2  0.8431454   86967.74
#0.4  2          0.6               0.50       100      140714.4  0.8536016   82038.79
#0.4  2          0.6               0.50       150      138213.0  0.8587548   79506.58
#0.4  2          0.6               0.75        50      146015.3  0.8420146   86279.94
#0.4  2          0.6               0.75       100      138949.4  0.8572695   80257.96
#0.4  2          0.6               0.75       150      134482.0  0.8666292   77126.52
#0.4  2          0.6               1.00        50      144913.1  0.8455197   85410.35
#0.4  2          0.6               1.00       100      135161.4  0.8656836   78618.73
#0.4  2          0.6               1.00       150      131422.3  0.8730928   75518.27
#0.4  2          0.8               0.50        50      145944.6  0.8433601   87098.59
#0.4  2          0.8               0.50       100      138678.4  0.8588376   81329.17
#0.4  2          0.8               0.50       150      136631.9  0.8635495   79261.61
#0.4  2          0.8               0.75        50      146604.5  0.8415342   86238.79
#0.4  2          0.8               0.75       100      139066.7  0.8576315   79867.99
#0.4  2          0.8               0.75       150      134366.9  0.8671912   76715.56
#0.4  2          0.8               1.00        50      143221.8  0.8481807   84488.49
#0.4  2          0.8               1.00       100      133793.4  0.8678417   77309.07
#0.4  2          0.8               1.00       150      130164.9  0.8747336   74484.79
#0.4  3          0.6               0.50        50      137503.2  0.8610904   79455.77
#0.4  3          0.6               0.50       100      135283.8  0.8656433   76458.23
#0.4  3          0.6               0.50       150      133508.7  0.8695067   74821.43
#0.4  3          0.6               0.75        50      136837.1  0.8614255   78983.53
#0.4  3          0.6               0.75       100      131145.0  0.8730181   74413.89
#0.4  3          0.6               0.75       150      129225.3  0.8767803   72595.35
#0.4  3          0.6               1.00        50      131858.0  0.8717864   76488.36
#0.4  3          0.6               1.00       100      127489.9  0.8800908   72170.45
#0.4  3          0.6               1.00       150      125531.9  0.8838704   70228.03
#0.4  3          0.8               0.50        50      137498.1  0.8618569   79186.59
#0.4  3          0.8               0.50       100      134096.1  0.8691465   75289.79
#0.4  3          0.8               0.50       150      134105.2  0.8694791   74401.46
#0.4  3          0.8               0.75        50      134438.0  0.8668021   78475.20
#0.4  3          0.8               0.75       100      130365.7  0.8749202   73799.16
#0.4  3          0.8               0.75       150      127959.3  0.8794646   71753.13
#0.4  3          0.8               1.00        50      130666.3  0.8739031   76403.02
#0.4  3          0.8               1.00       100      125536.7  0.8835805   71888.75
#0.4  3          0.8               1.00       150      123050.0  0.8879582   70064.24

#Tuning parameter 'gamma' was held constant at a value of 0
#Tuning parameter 'min_child_weight'
#was held constant at a value of 1
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were nrounds = 150, max_depth = 3, eta = 0.3, gamma =
#  0, colsample_bytree = 0.8, min_child_weight = 1 and subsample = 1.

xgbdat <- xgb.DMatrix(data = as.matrix(kcdat))
xgbt.model = xgboost(price~., data=xgbdat, nrounds = 150, max_depth = 3, eta = 0.3, gamma = 0, 
                     colsample_bytree = 0.8, min_child_weight = 1, subsample = 1)
plot(xgbt.model)



xgb.lr.all = train(price ~ ., data = kcdat, method = "xgbLinear",trControl = fitControl, metric="RMSE")

imp.xgb.lr.all <- varImp(xgb.lr.all, scale=FALSE)
plot(imp.xgb.lr.all)

#eXtreme Gradient Boosting 

#21613 samples
#20 predictor

#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 17291, 17290, 17291, 17290, 17290 
#Resampling results across tuning parameters:
  
#  lambda  alpha  nrounds  RMSE      Rsquared   MAE     
#0e+00   0e+00   50      128942.0  0.8775177  69221.23
#0e+00   0e+00  100      128414.6  0.8787169  68197.53
#0e+00   0e+00  150      128382.2  0.8788195  68215.46
#0e+00   1e-04   50      128942.0  0.8775177  69221.23
#0e+00   1e-04  100      128414.6  0.8787169  68197.53
#0e+00   1e-04  150      128382.2  0.8788195  68215.46
#0e+00   1e-01   50      128942.0  0.8775177  69221.23
#0e+00   1e-01  100      128414.6  0.8787169  68197.53
#0e+00   1e-01  150      128382.2  0.8788195  68215.46
#1e-04   0e+00   50      129029.0  0.8772773  69305.18
#1e-04   0e+00  100      128379.8  0.8786300  68436.09
#1e-04   0e+00  150      128618.0  0.8782845  68428.51
#1e-04   1e-04   50      129029.0  0.8772773  69305.18
#1e-04   1e-04  100      128379.8  0.8786300  68436.09
#1e-04   1e-04  150      128618.0  0.8782845  68428.51
#1e-04   1e-01   50      129029.0  0.8772773  69305.18
#1e-04   1e-01  100      128379.8  0.8786300  68436.09
#1e-04   1e-01  150      128619.1  0.8782836  68431.41
#1e-01   0e+00   50      127671.4  0.8794674  68993.76
#1e-01   0e+00  100      127136.6  0.8805735  68224.42
#1e-01   0e+00  150      127022.2  0.8808973  68150.77
#1e-01   1e-04   50      127671.4  0.8794674  68993.76
#1e-01   1e-04  100      127136.6  0.8805735  68224.42
#1e-01   1e-04  150      127022.2  0.8808973  68150.77
#1e-01   1e-01   50      127671.4  0.8794674  68993.76
#1e-01   1e-01  100      127136.6  0.8805735  68224.42
#1e-01   1e-01  150      127022.1  0.8808973  68150.77

#Tuning parameter 'eta' was held constant at a value of 0.3
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were nrounds = 150, lambda = 0.1, alpha = 0.1 and eta = 0.3.

#xgbl.model = xgboost(price~., data=xgbdat, nrounds = 150, lambda = 0.1, alpha = 0.1,
#                     eta = 0.3)
#plot(xgbl.model)



xgb.lr.all = train(price ~ ., data = kcdata, method = "xgbLinear",trControl = fitControl, metric="RMSE")
xgb.lrlog.all = train(log(price) ~ ., data = kcdat, method = "xgbLinear",trControl = fitControl, metric="RMSE")


############################## plot performance of xgboost tree and linear  ####################
samp <- sample(1:21613, size = 6484)
truep  <- as.matrix(kcdat[samp,1])
xgb_predicted = predict(xgb.lr.all, kcdat[samp,], "raw")
xgbt_predicted = predict(xgb.all, kcdat[samp,], "raw")

res <- data.frame(price=c(xgbt_predicted, xgb_predicted, truep),
                  type=c(replicate(length(xgbt_predicted), "xgb_linear"),
                         replicate(length(xgb_predicted), "xgb_tree"),
                         replicate(length(truep), "actual")
                  ))

ggplot(res, aes(x=price, colour=type)) +
  scale_x_continuous(limits = c(0,2e+06)) +
  scale_y_continuous() +
  geom_density()

######################### Feature Engineering ###############################
############################## dbscan #########################################
library(dbscan)
library(factoextra)
library(fpc)

dbscan::kNNdistplot(kcdat[,16:17])
abline(h = 0.01, lty = 2)

set.seed(123)
# fpc package
res.fpc <- fpc::dbscan(kcdat[,16:17], eps = 0.0114, MinPts = 50)
fviz_cluster(res.fpc, kcdat[,16:17], geom = "point")

# dbscan package
res.db <- dbscan::dbscan(kcdat, 0.0114, 50)

dbscan <- dbscan::dbscan(x = kcdat[, c('long', 'lat')], eps = 0.0114, minPts = 50)
fviz_cluster(dbscan, kcdat[, c('long', 'lat')], stand = FALSE,
             frame = FALSE, geom = "point", pointsize = 0.5)

kcdat1 <- cbind(kcdat, 'cluster' = as.factor(dbscan$cluster))
cluster_plot_A$cluster[cluster_plot_A$cluster == 0] <- NA


xgb.scan.all = train(price ~ ., data = kcdat1, method = "xgbTree",trControl = fitControl, metric="RMSE")

imp.xgb.scan.all <- varImp(xgb.scan.all, scale=FALSE)
plot(imp.xgb.scan.all)


############################ Splines ##########################################

# Regression Spline
reg_spline_clean <- lm(price~bs(bedrooms, df = 6)+
                         bs(bathrooms, df = 6)+
                         bs(sqft_living, df = 6)+
                         bs(sqft_lot, df = 6)+
                         bs(floors, df = 6)+
                         waterfront+
                         bs(view, df = 6)+
                         bs(condition, df = 6)+
                         bs(grade, df = 6)+
                         bs(sqft_above, df = 6)+
                         bs(yr_built, df = 6)+
                         yr_renovated+
                         zipcode+
                         bs(countcrime.y, df = 6)+
                         bs(Population, df = 6)
                       , data = kcdata)

train_reg_spline_clean <- train(price~bs(bedrooms, df = 6)+
                                  bs(bathrooms, df = 6)+
                                  bs(sqft_living, df = 6)+
                                  bs(sqft_lot, df = 6)+
                                  bs(floors, df = 6)+
                                  waterfront+
                                  bs(view, df = 6)+
                                  bs(condition, df = 6)+
                                  bs(grade, df = 6)+
                                  bs(sqft_above, df = 6)+
                                  bs(yr_built, df = 6)+
                                  yr_renovated+
                                  zipcode+
                                  bs(countcrime.y, df = 6)+
                                  bs(Population, df = 6),
                                  data = kcdata,"lm",
                                  trControl = trainControl(method="cv",number = 5))

# Natural Spline
nat_spline_clean <- lm(price~ns(bedrooms, df = 4)+
                         ns(bathrooms, df = 4)+
                         ns(sqft_living, df = 4)+
                         ns(sqft_lot, df = 4)+
                         ns(floors, df = 4)+
                         waterfront+
                         ns(view, df = 4)+
                         ns(condition, df = 4)+
                         ns(grade, df = 4)+
                         ns(sqft_above, knots = c(0.7, 0.8, 0.9))+
                         ns(yr_built, df = 4)+
                         yr_renovated+
                         zipcode
                         ns(countcrime.y, df = 4)+
                         ns(Population, df = 4),
                         data = kcdata)

train_nat_spline_clean <- train(price~ns(bedrooms, df = 4)+
                                  ns(bathrooms, df = 4)+
                                  ns(sqft_living, df = 4)+
                                  ns(sqft_lot, df = 4)+
                                  ns(floors, df = 4)+
                                  waterfront+
                                  ns(view, df = 4)+
                                  ns(condition, df = 4)+
                                  ns(grade, df = 4)+
                                  ns(sqft_above, knots = c(0.7, 0.8, 0.9))+
                                  ns(yr_built, df = 4)+
                                  yr_renovated+
                                  zipcode
                                  ns(countcrime.y, df = 4)+
                                  ns(Population, df = 4),
                                  data = kcdata,"lm",
                                  trControl = trainControl(method="cv",number = 5))

# Smoothing Spline
gam_all_clean <- gam(price~s(bedrooms)+
                       s(bathrooms)+
                       s(sqft_living)+
                       s(sqft_lot)+
                       floors+
                       waterfront+
                       view+
                       condition+
                       s(grade)+
                       s(sqft_above)+
                       s(yr_built)+
                       yr_renovated+
                       zipcode+
                       s(countcrime.y)+
                       s(Population),
                       data = kcdata)

# Plot
test1 <- lm(price~bs(bathrooms, knots = c(2,4,6)), data = kcdata)
test2 <- lm(price~ns(bathrooms, knots = c(2,4,6)), data = kcdata)
test3 <- smooth.spline(kcdata$bathrooms,kcdata$price,cv=TRUE)
k <- attr(bs(kcdata$bathrooms, knots = c(2,4,6)), "knots")

a<- sort(unique(kcdata$bathrooms))
pred1 <- predict(test1,newdata = list(bathrooms = a), se = TRUE)
pred2 <- predict(test2,newdata = list(bathrooms = a), se = TRUE)
pred3 <- predict(test3,newdata = list(bathrooms = a))

plot(kcdata$bathrooms, kcdata$price, col = "gray",
     xlab = 'Bathrooms', ylab = 'Price', main = 'Bathrooms vs Price')
lines(a, pred1$fit, lwd = 2, col = 'red')
lines(a, pred2$fit, lwd = 2, col = 'blue')
lines(a, pred3$y, lwd = 2, col = 'purple')
lines(a,pred1$fit+10*pred1$se, lty = 'dashed', col = 'red')
lines(a,pred1$fit-10*pred1$se, lty = 'dashed', col = 'red')
lines(a,pred2$fit+10*pred2$se, lty = 'dashed', col = 'blue')
lines(a,pred2$fit-10*pred2$se, lty = 'dashed', col = 'blue')
abline(v=k, lty = 'dashed')
legend("topleft", legend=c('Regression Splines','Natural splines', 'Smoothing splines'),
       col=c('red','blue', 'purple'), lty = 'solid', cex=0.5)



