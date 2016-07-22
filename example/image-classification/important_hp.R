.Libs()
library(caret)
library(randomForest)

ds0 <- fread("train_adam:L2-batch_size-lr-num_epochs-primary_metric-solver.csv")
ds1 <- fread("train_sgd:L2-batch_size-lr-num_epochs-primary_metric-solver.csv")
ds <- rbind(ds0, ds1)
X <- ds[, -c("meta_params", "exp_id", "train.accuracy", "train.ce"), with=F]
X <- model.matrix(~.-1, X)
y <- ds$train.ce
# rm(ds); gc()

# rf.fit <- train(X, y)

idx <- sample(nrow(X), nrow(X) * 0.8)
rf.fit <- ranger(y ~ ., data.frame(x=X[idx,], y=y[idx]),
                importance = "impurity",
                write.forest = T)
y.pred <- predict(rf.fit, dat=data.frame(x=X[-idx,]))
barplot(prop.table(importance(rf.fit)))


# visualise metric ~ LR + mini.batch + solver
ggplot(ds, aes(x=num.mini.batch, y=train.ce, col=solver, group=exp_id)) + geom_line() + facet_wrap(~lr)

# visualise metric ~ LR + mini.batch + solver + (not important, L2 )
ggplot(ds, aes(x=num.mini.batch, y=train.ce, col=solver, group=exp_id)) + geom_line() + facet_grid(L2 ~ lr)
# visualise metric ~ LR + mini.batch + solver + (not important, batch_size )
ggplot(ds, aes(x=num.mini.batch, y=train.ce, col=solver, group=exp_id)) + geom_line() + facet_grid(batch_size ~ lr)

## this example shows the random forest can be used to guide
## visulisatin, helps to find out which one is the most important
## paramters. by looking at the plots, one can shrink the sample space
## for the important paramters, and then fine-tune it. what it doesn't
## improve however, is the irrelevant paramters.
