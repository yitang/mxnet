library(stringr)
library(data.table)

log.file <- commandArgs(TRUE)

extract_learning_metrics <- function(fpath) {
    exp.log <- readLines(fpath)
    s.tmp <- grep("Train-accuracy", exp.log, v=T)
    s.tmp2 <- str_extract_all(s.tmp, 'Train-accuracy=[0-9.]+')
    s.tmp3 <- str_extract_all(s.tmp2, '[0-9.]+')
    train.acc <- sapply(s.tmp3, function(x) as.numeric(x))

    s.tmp <- grep("Validation-accuracy", exp.log, v=T)
    s.tmp2 <- str_extract_all(s.tmp, 'Validation-accuracy=[0-9.]+')
    s.tmp3 <- str_extract_all(s.tmp2, '[0-9.]+')
    val.acc <- sapply(s.tmp3, function(x) as.numeric(x))


    s.tmp <- grep("Train-cross-entropy", exp.log, v=T)
    s.tmp2 <- str_extract_all(s.tmp, 'Train-cross-entropy=[0-9.]+')
    s.tmp3 <- str_extract_all(s.tmp2, '[0-9.]+')
    train.ce <- sapply(s.tmp3, function(x) as.numeric(x))

    s.tmp <- grep("Validation-cross-entropy", exp.log, v=T)
    s.tmp2 <- str_extract_all(s.tmp, 'Validation-cross-entropy=[0-9.]+')
    s.tmp3 <- str_extract_all(s.tmp2, '[0-9.]+')
    val.ce <- sapply(s.tmp3, function(x) as.numeric(x))


    train.acc <- data.table(train.accuracy = train.acc,
                           train.ce = train.ce,
                           num.mini.batch = 1:length(train.acc))

    val.acc <- data.table(val.accuracy = val.acc,
                         val.ce = val.ce,
                         num.epoch = 1:length(val.acc))    
    return(list(train = train.acc,
           validation = val.acc))
}

ds <- extract_learning_metrics(log.file)
fwrite(ds$validation, file="tmp_validation.csv")
fwrite(ds$train, file="tmp_train.csv")
