## Natural Langauge Processing

## Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

## Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
as.character(corpus[[1]])
as.character(corpus[[841]])

## Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))

## Encoding the target feature as as factor
dataset$Liked = dataset_original$Liked
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

## Splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
set.seed(46)
split = sample.split(dataset$Liked, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## Creating the Random Forest Classification model
#install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

## Predicting the test set results
y_pred = predict(classifier, newdata = test_set[-692])

## Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
cm
