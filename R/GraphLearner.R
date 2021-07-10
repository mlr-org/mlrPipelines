#' @title Encapsulate a Graph as a Learner
#'
#' @name mlr_learners_graph
#' @format [`R6Class`] object inheriting from [`mlr3::Learner`].
#'
#' @description
#' A [`Learner`][mlr3::Learner] that encapsulates a [`Graph`] to be used in
#' [mlr3][mlr3::mlr3-package] resampling and benchmarks.
#'
#' The Graph must return a single [`Prediction`][mlr3::Prediction] on its `$predict()`
#' call. The result of the `$train()` call is discarded, only the
#' internal state changes during training are used.
#'
#' The `predict_type` of a [`GraphLearner`] can be obtained or set via it's `predict_type` active binding.
#' Setting a new predict type will try to set the `predict_type` in all relevant
#' [`PipeOp`] / [`Learner`][mlr3::Learner] encapsulated within the [`Graph`].
#' Similarly, the predict_type of a Graph will always be the smallest denominator in the [`Graph`].
#'
#' In the `timings_pipeops` active binding, the `timings` of the individual [`PipeOp`]s are stored as a named list
#' with names `"train"` and `"predict"`.
#' @family Learners
#' @export
GraphLearner = R6Class("GraphLearner", inherit = Learner,
  public = list(
    graph = NULL,
    initialize = function(graph, id = NULL, param_vals = list(), task_type = NULL, predict_type = NULL) {

      graph = as_graph(graph, clone = TRUE)

      # set the encapsulate of all pipeops to "none"
      for (i in seq_along(graph$pipeops)) {
        graph$pipeops[[i]]$encapsulate = c(train = "none", predict = "none")
      }

      id = assert_string(id, null.ok = TRUE) %??% paste(graph$ids(sorted = TRUE), collapse = ".")
      self$graph = graph
      output = graph$output
      if (nrow(output) != 1) {
        stop("'graph' must have exactly one output channel")
      }
      if (!are_types_compatible(output$predict, "Prediction")) {
        stop("'graph' output type not 'Prediction' (or compatible with it)")
      }

      if (is.null(task_type)) {
        # check the high level input and output
        class_table = mlr_reflections$task_types
        input = graph$input
        task_type = c(
          match(c(output$train, output$predict), class_table$prediction),
          match(c(input$train, input$predict), class_table$task))
        task_type = unique(class_table$type[stats::na.omit(task_type)])
        if (length(task_type) > 1L) {
          stopf("GraphLearner can not infer task_type from given Graph\nin/out types leave multiple possibilities: %s", str_collapse(task_type))
        }
        if (length(task_type) == 0L) {
          # recursively walk backwards through the graph
          get_po_task_type = function(x) {
            task_type = c(
              match(c(x$output$train, x$output$predict), class_table$prediction),
              match(c(x$input$train, x$input$predict), class_table$task))
            task_type = unique(class_table$type[stats::na.omit(task_type)])
            if (length(task_type) > 1) {
              stopf("GraphLearner can not infer task_type from given Graph\nin/out types leave multiple possibilities: %s", str_collapse(task_type))
            }
            if (length(task_type) == 1) {
              return(task_type)  # early exit
            }
            prdcssrs = graph$edges[dst_id == x$id, ]$src_id
            if (length(prdcssrs)) {
              # all non-null elements
              task_types = discard(map(graph$pipeops[prdcssrs], get_po_task_type), is.null)
              if (length(unique(task_types)) == 1L) {
                return(unlist(unique(task_types)))
              }
            }
            return(NULL)
          }
          task_type = get_po_task_type(graph$pipeops[[graph$rhs]])
        }
        task_type = c(task_type, "classif")[1L]  # final fallback
      }
      assert_subset(task_type, mlr_reflections$task_types$type)

      super$initialize(id = id, task_type = task_type,
        feature_types = mlr_reflections$task_feature_types,
        predict_types = names(mlr_reflections$learner_predict_types[[task_type]]),
        packages = graph$packages,
        properties = mlr_reflections$learner_properties[[task_type]])

      if (length(param_vals)) {
        self$graph$param_set$values = insert_named(self$graph$param_set$values, param_vals)
      }
      if (!is.null(predict_type)) self$predict_type = predict_type
    }
  ),
  active = list(
    hash = function() {
      digest(list(self$id, self$graph$hash), algo = "xxhash64")
    },
    predict_type = function(rhs) {
      if (!missing(rhs)) {
        private$set_predict_type(rhs)
      }
      private$get_predict_type()
    },
    param_set = function(rhs) {
      if (!missing(rhs) && !identical(rhs, self$graph$param_set)) {
        stop("param_set is read-only.")
      }
      self$graph$param_set
    },
    timings_pipeops = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(self$state$model)) {
        timing = stats::setNames(rep(NA_real_, length(self$graph$pipeops)), nm = names(self$graph$pipeops))
        return(list(train = timing, predict = timing))  # early exit
      }
      # reorder based on topologically sorted ids
      list(train = stats::setNames(map_dbl(self$state$model, function(pipeop) pipeop$train_time %??% NA_real_), nm = names(self$graph$pipeops))[self$graph$ids(TRUE)],
        predict = stats::setNames(map_dbl(self$state$model, function(pipeop) pipeop$predict_time %??% NA_real_), nm = names(self$graph$pipeops))[self$graph$ids(TRUE)])
    }
  ),
  private = list(
    deep_clone = function(name, value) {
      # FIXME: this repairs the mlr3::Learner deep_clone() method which is broken.
      if (is.environment(value) && !is.null(value[[".__enclos_env__"]])) {
        return(value$clone(deep = TRUE))
      }
      value
    },

    .train = function(task) {
      on.exit({self$graph$state = NULL})
      self$graph$train(task)
      state = self$graph$state
      state
    },
    .predict = function(task) {
      on.exit({self$graph$state = NULL})
      self$graph$state = self$model
      prediction = self$graph$predict(task)
      self$state$model = self$graph$state  # needed to get each pipeop's predict_time in the state
      assert_list(prediction, types = "Prediction", len = 1,
        .var.name = sprintf("Prediction returned by Graph %s", self$id))
      prediction[[1]]
    },
    get_predict_type = function() {
      # recursively walk backwards through the graph
      get_po_predict_type = function(x) {
        if (!is.null(x$predict_type)) return(x$predict_type)
        prdcssrs = self$graph$edges[dst_id == x$id, ]$src_id
        if (length(prdcssrs)) {
          # all non-null elements
          predict_types = discard(map(self$graph$pipeops[prdcssrs], get_po_predict_type), is.null)
          if (length(unique(predict_types)) == 1L)
            return(unlist(unique(predict_types)))
        }
        return(NULL)
      }
      predict_type = get_po_predict_type(self$graph$pipeops[[self$graph$rhs]])
      if (is.null(predict_type))
        mlr_reflections$learner_predict_types[[self$task_type]][[1]]
      else
        predict_type
    },
    set_predict_type = function(predict_type) {
      # recursively walk backwards through the graph
      set_po_predict_type = function(x, predict_type) {
        assert_subset(predict_type, unlist(mlr_reflections$learner_predict_types[[self$task_type]]))
        if (!is.null(x$predict_type)) x$predict_type = predict_type
        prdcssrs = self$graph$edges[dst_id == x$id, ]$src_id
        if (length(prdcssrs)) {
          map(self$graph$pipeops[prdcssrs], set_po_predict_type, predict_type = predict_type)
        }
      }
      set_po_predict_type(self$graph$pipeops[[self$graph$rhs]], predict_type)
    }
  )
)

#' @export
as_learner.Graph = function(x, clone = FALSE) {
  GraphLearner$new(x)
}
