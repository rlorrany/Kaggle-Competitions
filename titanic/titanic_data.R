# About the problem
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

library(replyr)
library(descr)
library(randomForest)
library(rpart)
library(party)
library(ROCR)
library(Amelia)
library(readr)
library(partykit)
library(caTools)
library(caret)
library(dplyr)
library(corrplot)
library(xgboost)
library(plyr)
library(LiblineaR)

# --------------------------------read data----------------------------------------------#
path <- "C:/Users/rhais/Documents/Rhaissa/Estudos/kaggle/titanic"
titanic <- read.csv(paste(path,"/originais/train.csv", sep = ''), header = TRUE, sep=',', stringsAsFactors=FALSE, na.strings=c(""," ","NA"))
titanic_test <- read.csv(paste(path,"/originais/test.csv", sep = ''), header = TRUE, sep=',', stringsAsFactors=FALSE)
View(head(titanic))
View(head(titanic_test))
# ------------------------------ split train and test -----------------------------------#

set.seed(-91101131)
sample <- sample.int(n = nrow(titanic), size = floor(.75*nrow(titanic)), replace = F)
train <- titanic[sample, ]
test  <- titanic[-sample, ]
 
# ------------------------------------------------------------------------------#
# Cleaning process
# 
# ------------------------------------------------------------------------------#

data = train
# ----- 1.1 check missing in all data with complete cases
n_complete = sum (complete.cases(data))
n_rows = nrow(data)
p_complete = n_complete /  n_rows
p_complete

# ----- 1.2 missing and empty cells in each variable
m = apply(is.na(data),2,sum) #check age - 132 NA's
m = t(t(m))
m <- as.data.frame(m)
colnames(m)[1]="qtd_miss"
m$prop_miss <- round(m$qtd_miss/nrow(data),2)
n_miss <- data.frame(var = row.names(m), n_miss = m$qtd_miss, prop_miss = m$prop_miss)
write.table(n_miss, paste(path,"/missing_variables.csv",sep=''), sep=';',col.names = T,
            row.names = F, dec=',')

# agrup "Age" because of 20%  missing data

# ----- 2. Treating Numerical 
num_vars <- c(names(which(sapply(train, class) == "numeric")), 
              names(which(sapply(train, class) == "integer" )))
num_vars

#Correlation Analysis
exclude_cor <- c("Age", "Survived","PassengerId") #because of missing
corr <- cor(data[,colnames(data)%in% num_vars & !colnames(data) %in% exclude_cor])
#corrplot(corr, order = "AOE", col = col2(50))
corrplot(corr, method = "number", order = "AOE")

# Categorical variables masquerading as numbers
ignore <- c("Pclass") # response variable and categorical variables maquerading as numbers
response <- c("Survived")
ignore_id <- c("PassengerId") # usually data has ID's

data[,response] <- as.factor(data[,response])
num_vars <- num_vars[!num_vars %in% c(ignore, ignore_id, response)] # vars to treat
num_vars

#treating missing
data["Age_2"] <- data["Age"]
data["Age_2"][is.na(data["Age_2"])] <- mean(data[,"Age"],na.rm = T)

ggplot(data) + 
  geom_density( aes_string(x = "Age", colour = "Survived"), size=0.8) +
  geom_density( aes_string(x = "Age_2", colour = "Survived"),linetype = 2,size = 0.9) +
  ggtitle( "Age" )+ theme(plot.title = element_text(lineheight=.11, face="bold", hjust = 0.5))


# Exploring data
# distribution and  boxplots
num_vars <- c(num_vars,"Age_2")

for (i in 1:length(num_vars)) {
  mypath_dens <- file.path(paste(path,"/figuras/density_", num_vars[i], ".jpg", sep = ""))
  mypath_box <- file.path(paste(path,"/figuras/boxplot_", num_vars[i], ".jpg", sep = ""))
  
  ggplot(data, aes_string(x = num_vars[i], colour = response)) + geom_density() +
  ggtitle(num_vars[i]) + theme(plot.title = element_text(lineheight=.11, face="bold", hjust = 0.5))
  ggsave(mypath_dens)
   
  ggplot(data, aes_string(x = response, y = num_vars[i],  colour = response)) +
    geom_boxplot() + ggtitle(num_vars[i]) +
    theme(plot.title = element_text(lineheight=.11, face="bold", hjust = 0.5))
 
  ggsave(mypath_box)

}




list_outliers <- list()
summary <- c()
for (i in 1:length(num_vars)) { 
  list_outliers[[i]] <- boxplot.stats(data[,num_vars[i]])$out 
  aux = as.vector(summary(data[,num_vars[i]])[1:6])
  summary = rbind(summary, aux)
}
colnames(summary) <- c("Min.","P25","P50","Mean","P75","Max")
summary <- as.data.frame(summary)
rownames(summary) <- num_vars
write.table(summary, paste(path,"/summary_numericvars.csv",sep=''), sep=';',
            dec=',', col.names = T, row.names =T)
View(summary)

# ---- 4. creating groups 
# quantiles
# intuitive

# ---- Numerical
# Age, Fare, SibSp, Parch
# split considering the variable response

crosstable <- table(data$Age, data$Survived)
write.table (crosstable, paste(path,"/crossage.csv",sep=''), sep=';')

data2 <- data %>%
  mutate (Age_c0 = ifelse(is.na(Age), c("missing/demais"), ifelse (Age<=6,c("1.Ate 6"),
                        ifelse(Age > 50,c("4. >50"),c("missing/demais"))))) 

CrossTable (data2[,"Age_c0"], data2[,response])

# quantiles - fast way to split in groups
# n_fx -> number of quantiles; if the data contains NA's, it will be replaced by 0s. After this, 
# the variable will be changed to factors.
# The new variable name: old_name + '_c'

names_var_quantile = num_vars[c(1:2)]
n_fx = c(5,8) #n_fx = rep(5,2) - same number for all variables
list_cross = list()

for (i in 1:length(names_var_quantile)) {

  new_var = paste(names_var_quantile[i],"_c",sep='')
  
  data2[,new_var] <- ntile(data2[,names_var_quantile[i]], n_fx[i])
  data2[new_var][is.na(data2[new_var])] <- 0
  data2[,new_var] <- as.factor(data2[,new_var])
  list_cross[[i]] <- CrossTable (data2[,new_var], data2[,response])
  # do the table with cut values  
}

data2[,"Fare_c2"] <- ifelse (as.numeric(data2[,"Fare_c"]) <= 3, 1, ifelse(as.numeric(data2[,"Fare_c"]) <= 7, 2, 3))


# Numerical: SibSp e / Parch
CrossTable (data2[,"SibSp"], data2[,response])
CrossTable (data2[,"Parch"], data2[,response])

data2[,"SibSp_c"] <- ifelse(data2[,"SibSp"] == 0,"0",ifelse (data2[,"SibSp"] >= 3, ">=3",
                    ifelse(data2[,"SibSp"] == 1 | data2[,"SibSp"] == 2, "1 or 2",">=3")))
data2[,"Parch_c"] <- ifelse(data2[,"Parch"] == 1,"1", ifelse(data2[,"Parch"] == 2, "2", 
                     ifelse(data2[,"Parch"] == 3, "3", 
                     ifelse(data2[,"Parch"] == 0 | data2[,"Parch"] > 3, "0 ou >3","0 ou >3"))))

CrossTable (data2[,"SibSp_c"], data2[,response])
CrossTable (data2[,"Parch_c"], data2[,response])


# SibSp - 0 (-)/ 1,2 (+)/ >=3 (-)
# Parch - 1,2,3 (+)

#--------------------------------------------------------------------#
# Categorical: Names, Embarked, Cabin, Ticket
char_vars <- c(names(which(sapply(train, class) == "character")))
ignore2 <- c("Name")
char_vars <- char_vars[!char_vars %in% c(ignore2)] # vars to treat

data2$Name_c <- ifelse(regexpr("Mrs.", data2[,"Name"]) > 0, "Mrs.",
                       ifelse(regexpr("Miss.", data2[,"Name"]) > 0, "Miss.",
                              ifelse(regexpr("Mr.", data2[,"Name"]) > 0, "Mr.",
                                            ifelse(regexpr("Master.", data2[,"Name"]) > 0, "Master.",
                                                   "Others"))))
summary_cat<-c()
list_cross2 <- list()
for (i in 1:length(char_vars)) {
  list_cross2[[i]] <- CrossTable (data2[,char_vars[i]], data2[,response])
  aux = length(unique(data2[,char_vars[i]]))
  summary_cat = rbind(summary_cat, aux)
} 
rownames(summary_cat) = char_vars  
colnames(summary_cat) <- c("n_levels")
summary_cat <- as.data.frame(summary_cat)
write.table(summary_cat, paste(path,"/summary_charvars.csv",sep=''), sep=';',
            dec=',', col.names = T, row.names =T)



#embarked
data2$Embarked[is.na(data2$Embarked)] <- "empty"
data2$Embarked_c <- ifelse(data2$Embarked == "empty","S",data2$Embarked)

#too many categorys: ticket and cabin
View(data2[,c("Ticket","Cabin")])

#cabin -
data2$Cabin[is.na(data2$Cabin)] <- "empty"
data2$Cabin_str <- substr(data2[,"Cabin"],1,1)
data2$Cabin_c <- ifelse(data2$Cabin == "empty", "empty",
                 ifelse(data2$Cabin_str == "D" | data2$Cabin_str == "E", "D/E",
                 ifelse(data2$Cabin_str == "G" | data2$Cabin_str == "T", "G/T/NA",
                 ifelse(data2$Cabin_str == "F" | data2$Cabin_str == "C", "C/F",
                 ifelse(data2$Cabin_str == "A", "A",
                 ifelse(data2$Cabin_str == "B", "B","G/T/NA"))))))

CrossTable(data2[,"Cabin_c"], data2[,response])

# Ticket
View(data2[,c("Ticket","Ticket_c")])
data2$Ticket_c <- ifelse(regexpr("C.A.", data2[,"Ticket"]) > 0 |
                           regexpr("CA", data2[,"Ticket"]) > 0, "C.A.",
                  ifelse(regexpr("PC", data2[,"Ticket"]) > 0, "PC",
                  ifelse(regexpr("STON", data2[,"Ticket"]) > 0, "STON",
                  ifelse(regexpr("SOTON", data2[,"Ticket"]) > 0, "STON",
                  ifelse(regexpr("C", data2[,"Ticket"]) > 0, "Letters",
                  ifelse(regexpr("P", data2[,"Ticket"]) > 0, "Letters",
                  ifelse(regexpr("A", data2[,"Ticket"]) > 0, "Letters",
                  ifelse(regexpr("W", data2[,"Ticket"]) > 0, "Letters",
                  ifelse(regexpr("S", data2[,"Ticket"]) > 0, "Letters",
                            "Numbers")))))))))

CrossTable(data2[,"Ticket_c"], data2[,response])


#Exploration data
# vars
exclude_chr <- c("Ticket","Cabin")

char_vars_complete <- char_vars[!char_vars %in% exclude_chr] # vars to treat

char_vars_complete <- c(char_vars_complete, "Ticket_c","Cabin_c","Name_c","Fare_c2","Age_c0",
                        "Parch_c",  "SibSp_c" , "Pclass","Parch", "SibSp")

chr_response <- "1"
  
# table
tables_chr_vars=list()
#i=1
for (i in 1:length(char_vars_complete)) {
  
  tab = table(data2[,char_vars_complete[i]], data2[,response])
  sum_cat = apply(tab,1,sum)
  p = round(tab[,chr_response]/sum_cat,2)
  cat = row.names(tab)
  final = cbind(sum_cat,p)
  final = as.data.frame(final)
  final$cat = cat
  tables_chr_vars[[i]] = final
  

#plot
  mypath_bar <- file.path(paste(path,"/figuras/barplot_", char_vars_complete[i], ".jpg", sep = ""))
  
  ggplot (data = final) +
    geom_bar(aes(x = factor(cat),y = sum_cat), stat="identity", fill = "turquoise3",width=.4) +
    geom_line(data = final, aes(x = factor(cat), y = p*max(final$sum_cat),group=1), size=1.5,
              stat = "identity",colour ="#DD8888")+
    geom_text(aes(label=p, x=factor(cat), y=p*max(final$sum_cat)), colour="black") +
    scale_y_continuous(sec.axis = sec_axis(~./max(final$sum_cat), name = "% Survived")) +
    ggtitle(char_vars_complete[i]) +
    theme(plot.title = element_text(lineheight=.11, face="bold", hjust = 0.5))+
    xlab("") +
    ylab("count")
  
  ggsave(mypath_bar)
}




# factors after making groups again
names_to_fac <- c("Name_c", "Fare_c2","Pclass","Ticket_c","Cabin_c","Sex",
                  "SibSp_c", "Parch_c","Age_c0","Embarked_c")
#names_to_chr <- c("Name_c", "Fare_c2","Pclass","Ticket_c","Cabin_c","Sex",
#                  "SibSp_c", "Parch_c","Age_c0","Embarked")
data2[,names_to_fac] <- lapply(data2[,names_to_fac], factor)
#data2[,names_to_chr] <- lapply(data2[,names_to_chr], as.character)


View(head(data2))
 
#----------------------------------------------------------------#
levels <- unique(data2$Survived)
data2$Survived <- factor(data2$Survived,labels=make.names(levels))

x_train = data2[,colnames(data2) %in% c(names_to_fac, "Fare","Age_2")]
y_train = data2$Survived
#------------------------------------------------------------------------#
#Modeling
#------------------------------------------------------------------------#

fitControl <- trainControl(method = "repeatedcv",
                           number = 8, 
                           repeats = 20,
                           savePredictions = "final")
# ----------------------------- Decision tree ---------------------------#

set.seed(18225)
tree <- train(x = x_train, y = y_train,  
            method = "rpart2", 
            trControl = fitControl, 
            tuneGrid = expand.grid(maxdepth = c(1:15)))
#trellis.par.set(caretTheme())
plot(tree) 

# 
# ------------------------------ Random Forest --------------------------#
t_inicio = Sys.time()
set.seed(-125)
rf <- train (x = x_train, y = y_train,  
                 method = "rf", 
                 trControl = fitControl, 
              #   verbose = FALSE, 
                 ntree = 200,
                 tuneGrid = expand.grid(mtry = c(2:10)))
            
plot(rf) 
t_fim = Sys.time()
t_fim-t_inicio
# ------------------------------ xgboost --------------------------#
t_inicio = Sys.time()
set.seed(-81250)


svm <- train (x = x_train, y = y_train,  
             method = "svmLinear3", 
            # verbose = TRUE,
             trControl = fitControl, 
            tuneLength = 2)

plot(xgboost) 
t_fim = Sys.time()
t_fim-t_inicio