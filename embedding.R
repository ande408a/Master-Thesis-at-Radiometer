train=read.csv("complaints_train.csv")
test=read.csv("complaints_test.csv")
library(keras)

y_train=train$product
y_train=as.factor(train$product)
y_train=as.integer(y_train)
y_train=y_train-1
x_train=train$narrative

y_test=test$product
y_test=as.factor(test$product)
y_test=as.integer(y_test)
y_test=y_test-1
x_test=test$narrative

num_words <- 1000

tokenizer <- text_tokenizer(
  num_words = num_words,
  filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
  lower = TRUE,
  split = " ",
  char_level = FALSE,
  oov_token = NULL
) 

tokenizer %>% fit_text_tokenizer(x_train)


word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

sequences <- texts_to_sequences(tokenizer, x_train)
sequencest <- texts_to_sequences(tokenizer, x_test)
sequences %>% head()
sequencest %>% head()

maxlen <- 25
x_train <- pad_sequences(sequences, maxlen = maxlen)
x_test = pad_sequences(sequencest, maxlen = maxlen)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = num_words,
                  output_dim = 50,
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  verbose =1
)

plot(history)


model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = num_words,
                  output_dim = 50,
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 4,
  batch_size = 32,
  verbose =1
)

plot(history)