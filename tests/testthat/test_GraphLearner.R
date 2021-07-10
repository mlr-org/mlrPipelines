context("GraphLearner")

test_that("basic graphlearner tests", {
  skip_on_cran()  # takes too long
  task = mlr_tasks$get("iris")

  lrn = mlr_learners$get("classif.rpart")
  gr = PipeOpLearner$new(lrn)

  glrn = GraphLearner$new(gr)
  expect_true(run_experiment(task, glrn)$ok)

  glrn = GraphLearner$new(gr)
  expect_learner(glrn)
  glrn$train(task)

  expect_prediction_classif({
    graphpred = glrn$predict(task)
  })
  expect_equal(graphpred,
    lrn$train(task)$predict(task))

  expect_prediction_classif({
    graphpred = glrn$predict(task)
  })
  expect_equal(graphpred, lrn$predict(task))

  set.seed(1)
  lrn = mlr_learners$get("classif.rpart")
  resgraphlrn = resample(task, lrn, mlr_resamplings$get("cv"))
  set.seed(1)
  resjustlrn = resample(task, lrn, mlr_resamplings$get("cv"))
  expect_equal(resgraphlrn$prediction(), resjustlrn$prediction())

  gr2 = PipeOpScale$new() %>>% PipeOpLearner$new(lrn)
  glrn2 = GraphLearner$new(gr2)
  glrn2$graph$set_names(c("scale", "classif.rpart"), new = c("newscale", "newclassif.rpart"))
  expect_list(glrn2$timings_pipeops, len = 2L)
  expect_equal(names(glrn2$timings_pipeops), c("train", "predict"))
  expect_equal(names(glrn2$timings_pipeops$train), glrn2$graph$ids(TRUE))
  expect_equal(names(glrn2$timings_pipeops$train), names(glrn2$timings_pipeops$predict))
  expect_true(every(glrn2$timings_pipeops, function(phase) every(phase, is.na)))
  glrn2_clone = glrn2$clone(deep = TRUE)
  expect_learner(glrn2)
  expect_true(run_experiment(task, glrn)$ok)
  glrn2$train(task)
  expect_list(glrn2$timings_pipeops, len = 2L)
  expect_equal(names(glrn2$timings_pipeops), c("train", "predict"))
  expect_equal(names(glrn2$timings_pipeops$train), names(glrn2$timings_pipeops$predict))
  expect_true(every(is.na(glrn2$timings_pipeops$train), isFALSE))
  expect_true(every(is.na(glrn2$timings_pipeops$predict), isTRUE))
  glrn2_clone$state = glrn2$state
  # glrn2_clone$state$log = glrn2_clone$state$log$clone(deep = TRUE)  # FIXME: this can go when mlr-org/mlr3#343 is fixed
  # glrn2_clone$state$model$classif.rpart$log = glrn2_clone$state$model$classif.rpart$log$clone(deep = TRUE)  # FIXME: this can go when mlr-org/mlr3#343 is fixed
  expect_deep_clone(glrn2_clone, glrn2$clone(deep = TRUE))
  expect_prediction_classif({
    graphpred2 = glrn2$predict(task)
  })
  expect_list(glrn2$timings_pipeops, len = 2L)
  expect_equal(names(glrn2$timings_pipeops), c("train", "predict"))
  expect_equal(names(glrn2$timings_pipeops$train), glrn2$graph$ids(TRUE))
  expect_equal(names(glrn2$timings_pipeops$train), names(glrn2$timings_pipeops$predict))
  expect_true(every(is.na(glrn2$timings_pipeops$train), isFALSE))
  expect_true(every(is.na(glrn2$timings_pipeops$predict), isFALSE))

  expect_equal(glrn2$predict(task), glrn2_clone$predict(task))

  glrn2$graph$set_names(c("newscale", "newclassif.rpart"), new = c("NEWscale", "NEWclassif.rpart"))
  expect_equal(names(glrn2$timings_pipeops$train), glrn2$graph$ids(TRUE))
  expect_equal(names(glrn2$timings_pipeops$train), names(glrn2$timings_pipeops$predict))

  scidf = cbind(scale(iris[1:4]), iris[5])
  scalediris = TaskClassif$new("scalediris", as_data_backend(scidf), "Species")

  dblrn = mlr_learners$get("classif.debug")
  dblrn$param_set$values$save_tasks = TRUE

  dbgr = GraphLearner$new(PipeOpScale$new() %>>% PipeOpLearner$new(dblrn))


  expect_equal(dbgr$train(task), dbgr)

  # debuglearner predict() modifies model, but PipeOpLearner does not accept
  # model changes in predict phase, so would ordinarily discard the change.
  # Here we swap the debuglearner model by an environment, which gets updated
  # by-reference, so we can get the $task_predict slot eventually.
  dbmodels = as.environment(dbgr$model$classif.debug$model)
  dbgr$state$model$classif.debug$model = dbmodels

  dbgr$predict(task)

  expect_equal(dbmodels$task_train$data(), scalediris$data())
  expect_equal(dbmodels$task_predict$data(), scalediris$data())
})

test_that("graphlearner parameters behave as they should", {
  dblrn = mlr_learners$get("classif.debug")
  dblrn$param_set$values$save_tasks = TRUE

  dbgr = PipeOpScale$new() %>>% PipeOpLearner$new(dblrn)

  expect_subset(c("scale.center", "scale.scale", "classif.debug.x"), names(dbgr$param_set$params))

  dbgr$param_set$values$classif.debug.x = 1

  expect_equal(dbgr$param_set$values$classif.debug.x, 1)
  expect_equal(dbgr$pipeops$classif.debug$param_set$values$x, 1)
  expect_equal(dbgr$pipeops$classif.debug$learner$param_set$values$x, 1)

  dbgr$pipeops$classif.debug$param_set$values$x = 0

  expect_equal(dbgr$param_set$values$classif.debug.x, 0)
  expect_equal(dbgr$pipeops$classif.debug$param_set$values$x, 0)
  expect_equal(dbgr$pipeops$classif.debug$learner$param_set$values$x, 0)

  dbgr$pipeops$classif.debug$learner$param_set$values$x = 0.5

  expect_equal(dbgr$param_set$values$classif.debug.x, 0.5)
  expect_equal(dbgr$pipeops$classif.debug$param_set$values$x, 0.5)
  expect_equal(dbgr$pipeops$classif.debug$learner$param_set$values$x, 0.5)

  expect_error({
    dbgr$param_set$values$classif.debug.x = "a"
  })
  expect_error({
    dbgr$pipeops$classif.debug$param_set$values$x = "a"
  })
  expect_error({
    dbgr$pipeops$classif.debug$learner$param_set$values$x = "a"
  })

  expect_equal(dbgr$param_set$values$classif.debug.x, 0.5)
  expect_equal(dbgr$pipeops$classif.debug$param_set$values$x, 0.5)
  expect_equal(dbgr$pipeops$classif.debug$learner$param_set$values$x, 0.5)

  dblrn = mlr_learners$get("classif.debug")
  dblrn$param_set$values$message_train = 1
  dblrn$param_set$values$message_predict = 1
  dblrn$param_set$values$warning_train = 1
  dblrn$param_set$values$warning_predict = 1

  pol = PipeOpLearner$new(dblrn, param_vals = list(message_predict = 0, warning_train = 0, warning_predict = 0))

  gl = GraphLearner$new(pol, param_vals = list(classif.debug.warning_train = 1, classif.debug.warning_predict = 1))

  gl$param_set$values$classif.debug.warning_predict = 0

  expect_equal(gl$param_set$values,
    list(classif.debug.message_train = 1, classif.debug.message_predict = 0, classif.debug.warning_train = 1, classif.debug.warning_predict = 0))
})

test_that("graphlearner type inference", {
  skip_on_cran()  # takes too long
  # default: classif
  lrn = GraphLearner$new(mlr_pipeops$get("nop"))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  ###########
  # classif #
  ###########

  # inference from pipeoplearner
  lrn = GraphLearner$new(mlr_pipeops$get("learner", lrn("classif.rpart")))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  # inference from output only
  lrn = GraphLearner$new(mlr_pipeops$get("copy", 1) %>>% mlr_pipeops$get("learner", lrn("classif.rpart")))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  # inference from input only
  lrn = GraphLearner$new(mlr_pipeops$get("learner", lrn("classif.rpart")) %>>% mlr_pipeops$get("copy", 1))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  # inference when multiple input, but one is a Task
  # inference when multiple input, but one is a Task

  lrn = GraphLearner$new(gunion(list(mlr_pipeops$get("learner", lrn("regr.rpart")), mlr_pipeops$get("nop"))) %>>% mlr_pipeops$get("unbranch"))
  expect_equal(lrn$task_type, "regr")
  # expect_equal(lrn$predict_type, "response")

  ###########
  # Errors  #
  ###########

  # input, output mismatching types
  gr = gunion(list(mlr_pipeops$get("learner", lrn("regr.rpart")), mlr_pipeops$get("nop"))) %>>% mlr_pipeops$get("unbranch") %>>% mlr_pipeops$get("learner", lrn("classif.rpart"))
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  gr = gunion(list(mlr_pipeops$get("learner", lrn("classif.rpart")), mlr_pipeops$get("nop"))) %>>% mlr_pipeops$get("unbranch") %>>% mlr_pipeops$get("learner", lrn("regr.rpart"))
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  # input two mismatching types
  gr = gunion(list(mlr_pipeops$get("learner", lrn("classif.rpart")), mlr_pipeops$get("learner", lrn("regr.rpart")))) %>>% mlr_pipeops$get("unbranch")
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  # input two mismatching types
  expect_error(GraphLearner$new(PipeOpScale$new()), "output type not.*Prediction.*or compatible")

  ###########################
  # Target Transformations  #
  ###########################

  lrn = GraphLearner$new(ppl("targettrafo", graph = lrn("classif.rpart"), trafo_pipeop = PipeOpTargetMutate$new()))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  lrn = GraphLearner$new(ppl("targettrafo", graph = lrn("regr.rpart"), trafo_pipeop = PipeOpTargetMutate$new()))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")

  lrn = GraphLearner$new(ppl("targettrafo", graph = lrn("regr.rpart"), trafo_pipeop = PipeOpTargetTrafoScaleRange$new()))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")
})

test_that("graphlearner type inference - branched", {
  skip_on_cran()  # takes too long

  # default: classif

  lrn = GraphLearner$new(gunion(list(
      mlr_pipeops$get(id = "l1", "learner", lrn("classif.rpart")),
      po("nop") %>>% mlr_pipeops$get(id = "l2", "learner", lrn("classif.rpart"))

    )) %>>%
    po("classifavg") %>>%
    po(id = "n2", "nop"))
  expect_equal(lrn$task_type, "classif")
  expect_equal(lrn$predict_type, "response")

  ###########
  # regr    #
  ###########

  # inference from pipeoplearner
  lrn = GraphLearner$new(mlr_pipeops$get("learner", lrn("regr.rpart")))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")

  # inference from output only
  lrn = GraphLearner$new(mlr_pipeops$get("copy", 1) %>>% mlr_pipeops$get("learner", lrn("regr.rpart")))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")

  # inference from input only
  lrn = GraphLearner$new(mlr_pipeops$get("learner", lrn("regr.rpart")) %>>% mlr_pipeops$get("copy", 1))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")

  # inference when multiple input, but one is a Task
  lrn = GraphLearner$new(gunion(list(
      mlr_pipeops$get(id = "l1", "learner", lrn("regr.rpart")),
      po("nop") %>>% mlr_pipeops$get(id = "l2", "learner", lrn("regr.rpart"))
    )) %>>%
    po("regravg") %>>%
    po(id = "n2", "nop"))
  expect_equal(lrn$task_type, "regr")
  expect_equal(lrn$predict_type, "response")

  ###########
  # Errors  #
  ###########

  # input, output mismatching types
  gr = gunion(list(mlr_pipeops$get("learner", lrn("regr.rpart")), mlr_pipeops$get("nop"))) %>>% mlr_pipeops$get("unbranch") %>>% mlr_pipeops$get("learner", lrn("classif.rpart"))
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  gr = gunion(list(mlr_pipeops$get("learner", lrn("classif.rpart")), mlr_pipeops$get("nop"))) %>>% mlr_pipeops$get("unbranch") %>>% mlr_pipeops$get("learner", lrn("regr.rpart"))
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  # input two mismatching types
  gr = gunion(list(mlr_pipeops$get("learner", lrn("classif.rpart")), mlr_pipeops$get("learner", lrn("regr.rpart")))) %>>% mlr_pipeops$get("unbranch")
  expect_error(GraphLearner$new(gr), "multiple possibilities")

  # input two mismatching types
  expect_error(GraphLearner$new(PipeOpScale$new()), "output type not.*Prediction.*or compatible")

})

test_that("graphlearner predict type inference", {
  skip_on_cran()  # takes too long
  # Getter:

  # Classification
  lrp = po(lrn("classif.rpart", predict_type = "prob"))
  lrr = po(lrn("classif.rpart"))
  lfp = po(lrn("classif.featureless", predict_type = "prob"))
  lfr = po(lrn("classif.featureless"))
  nop = po("nop")

  # linear
  lrn = GraphLearner$new(lrp)
  expect_equal(lrn$predict_type, "prob")
  lrn = GraphLearner$new(lrr)
  expect_equal(lrn$predict_type, "response")
  lrn = GraphLearner$new(lrp %>>% nop)
  expect_equal(lrn$predict_type, "prob")

  # averager
  lrn = GraphLearner$new(pipeline_greplicate(po("subsample") %>>% lrr, 3L) %>>% po("classifavg"))
  expect_equal(lrn$predict_type, "response")
  lrn = GraphLearner$new(pipeline_greplicate(po("subsample") %>>% lrp, 3L) %>>% po("classifavg"))
  expect_equal(lrn$predict_type, "prob")

  # branching
  lrn = GraphLearner$new(po("branch", 2) %>>% gunion(list(lrp, lfp)) %>>% po("unbranch"))
  expect_equal(lrn$predict_type, "prob")
  lrn = GraphLearner$new(po("branch", 2) %>>% gunion(list(lrr, lfr)) %>>% po("unbranch"))
  expect_equal(lrn$predict_type, "response")
  lrn = GraphLearner$new(po("branch", 2) %>>% gunion(list(lrp, lfr)) %>>% po("unbranch"))
  expect_equal(lrn$predict_type, "response")

  # with additional NOP in branch
  lrn = GraphLearner$new(po("branch", 2) %>>% gunion(list(lrp %>>% nop, lfp)) %>>% po("unbranch"))
  expect_equal(lrn$predict_type, "prob")

  # Regression
  lrrp = po(lrn("regr.featureless", predict_type = "se"))
  lrrr = po(lrn("regr.rpart"))
  lrn = GraphLearner$new(pipeline_greplicate(po("subsample") %>>% lrrr, 3L) %>>% po("regravg"))
  expect_equal(lrn$predict_type, "response")
  lrn = GraphLearner$new(pipeline_greplicate(po("subsample") %>>% lrrp, 3L) %>>% po("regravg"))
  expect_equal(lrn$predict_type, "se")

  lrn = GraphLearner$new(lrrp %>>% nop)
  expect_equal(lrn$predict_type, "se")


  # Setter:
  lrp = po(lrn("classif.rpart", predict_type = "prob"))

  lrn = GraphLearner$new(lrp)
  lrn$predict_type = "prob"
  expect_equal(lrn$predict_type, "prob")
  expect_equal(lrn$graph$pipeops[[lrp$id]]$predict_type, "prob")

  lrn = GraphLearner$new(lrp)
  lrn$predict_type = "response"
  expect_equal(lrn$predict_type, "response")
  expect_equal(lrn$graph$pipeops[[lrp$id]]$predict_type, "response")

  lrn = GraphLearner$new(lrp)
  lrn$predict_type = "prob"
  expect_equal(lrn$predict_type, "prob")
  expect_equal(lrn$graph$pipeops[[lrp$id]]$predict_type, "prob")

  lrn = GraphLearner$new(lrp %>>% po("nop"))
  lrn$predict_type = "response"
  expect_equal(lrn$predict_type, "response")
  expect_equal(lrn$graph$pipeops[[lrp$id]]$predict_type, "response")

  # averager
  lrn = GraphLearner$new(pipeline_greplicate(po("subsample") %>>% lrp %>>% nop, 3L) %>>% po("classifavg"))
  lrn$predict_type = "response"
  expect_equal(lrn$predict_type, "response")
  expect_true(all(map_chr(lrn$graph$pipeops[paste(lrp$id, 1:3, sep = "_")], "predict_type") == "response"))

  # branching
  lrn = GraphLearner$new(po("branch", 2) %>>% gunion(list(lrp, lfp %>>% nop)) %>>% po("unbranch"))
  expect_equal(lrn$predict_type, "prob")
  lrn$predict_type = "response"
  expect_equal(lrn$predict_type, "response")
  expect_equal(lrn$graph$pipeops[[lrp$id]]$predict_type, "response")
  expect_equal(lrn$graph$pipeops[[lfp$id]]$predict_type, "response")

  # Setter on construction
  lrn = GraphLearner$new(lrr, predict_type = "prob")
  expect_equal(lrr$predict_type, "response")
  expect_equal(lrn$predict_type, "prob")
  expect_equal(lrn$graph$pipeops[[lrr$id]]$predict_type, "prob")

  # Errors:
  expect_error({lrrp = po(lrn("classif.featureless", predict_type = "se"))})
})
