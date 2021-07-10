#' @title Wrap a Learner into a PipeOp
#'
#' @usage NULL
#' @name mlr_pipeops_learner
#' @format [`R6Class`] object inheriting from [`PipeOp`].
#'
#' @description
#' Wraps an [`mlr3::Learner`] into a [`PipeOp`].
#'
#' Inherits the `$param_set` (and therefore `$param_set$values`) from the [`Learner`][mlr3::Learner] it is constructed from.
#'
#' Using [`PipeOpLearner`], it is possible to embed [`mlr3::Learner`]s into [`Graph`]s, which themselves can be
#' turned into Learners using [`GraphLearner`]. This way, preprocessing and ensemble methods can be included
#' into a machine learning pipeline which then can be handled as singular object for resampling, benchmarking
#' and tuning.
#'
#' @section Construction:
#' ```
#' PipeOpLearner$new(learner, id = NULL, param_vals = list())
#' ```
#'
#' * `learner` :: [`Learner`][mlr3::Learner] | `character(1)`
#'   [`Learner`][mlr3::Learner] to wrap, or a string identifying a [`Learner`][mlr3::Learner] in the [`mlr3::mlr_learners`] [`Dictionary`][mlr3misc::Dictionary].
#' * `id` :: `character(1)`
#'   Identifier of the resulting  object, internally defaulting to the `id` of the [`Learner`][mlr3::Learner] being wrapped.
#' * `param_vals` :: named `list`\cr
#'   List of hyperparameter settings, overwriting the hyperparameter settings that would otherwise be set during construction. Default `list()`.
#'
#' @section Input and Output Channels:
#' [`PipeOpLearner`] has one input channel named `"input"`, taking a [`Task`][mlr3::Task] specific to the [`Learner`][mlr3::Learner]
#' type given to `learner` during construction; both during training and prediction.
#'
#' [`PipeOpLearner`] has one output channel named `"output"`, producing `NULL` during training and a [`Prediction`][mlr3::Prediction] subclass
#' during prediction; this subclass is specific to the [`Learner`][mlr3::Learner] type given to `learner` during construction.
#'
#' The output during prediction is the [`Prediction`][mlr3::Prediction] on the prediction input data, produced by the [`Learner`][mlr3::Learner]
#' trained on the training input data.
#'
#' @section State:
#' The `$state` is set to the `$state` slot of the [`Learner`][mlr3::Learner] object. It is a named `list` with members:
#' * `model` :: `any`\cr
#'   Model created by the [`Learner`][mlr3::Learner]'s `$.train()` function.
#' * `train_log` :: [`data.table`] with columns `class` (`character`), `msg` (`character`)\cr
#'   Errors logged during training.
#' * `train_time` :: `numeric(1)`\cr
#'   Training time, in seconds.
#' * `predict_log` :: `NULL` | [`data.table`] with columns `class` (`character`), `msg` (`character`)\cr
#'   Errors logged during prediction.
#' * `predict_time` :: `NULL` | `numeric(1)`
#'   Prediction time, in seconds.
#'
#' @section Parameters:
#' The parameters are exactly the parameters of the [`Learner`][mlr3::Learner] wrapped by this object.
#'
#' @section Internals:
#' The `$state` is currently not updated by prediction, so the `$state$predict_log` and `$state$predict_time` will always be `NULL`.
#'
#' @section Fields:
#' Fields inherited from [`PipeOp`], as well as:
#' * `learner` :: [`Learner`][mlr3::Learner]\cr
#'   [`Learner`][mlr3::Learner] that is being wrapped. Read-only.
#' * `learner_model` :: [`Learner`][mlr3::Learner]\cr
#'   [`Learner`][mlr3::Learner] that is being wrapped. This learner contains the model if the `PipeOp` is trained. Read-only.
#'
#' @section Methods:
#' Methods inherited from [`PipeOp`].
#'
#' @family PipeOps
#' @family Meta PipeOps
#' @seealso https://mlr3book.mlr-org.com/list-pipeops.html
#' @include PipeOp.R
#' @export
#' @examples
#' library("mlr3")
#'
#' task = tsk("iris")
#' learner = lrn("classif.rpart", cp = 0.1)
#' lrn_po = mlr_pipeops$get("learner", learner)
#'
#' lrn_po$train(list(task))
#' lrn_po$predict(list(task))
PipeOpLearner = R6Class("PipeOpLearner", inherit = PipeOp,
  public = list(
    initialize = function(learner, id = NULL, param_vals = list()) {
      private$.learner = as_learner(learner, clone = TRUE)
      id = id %??% private$.learner$id
      # FIXME: can be changed when mlr-org/mlr3#470 has an answer
      task_type = mlr_reflections$task_types[get("type") == private$.learner$task_type][order(get("package"))][1L]$task
      out_type = mlr_reflections$task_types[get("type") == private$.learner$task_type][order(get("package"))][1L]$prediction
      super$initialize(id, param_set = alist(private$.learner$param_set), param_vals = param_vals,
        input = data.table(name = "input", train = task_type, predict = task_type),
        output = data.table(name = "output", train = "NULL", predict = out_type),
        tags = "learner"
      )
    }
  ),
  active = list(
    id = function(val) {
      if (!missing(val)) {
        private$.id = val
        private$.learner$param_set$set_id = val
      }
      private$.id
    },
    learner = function(val) {
      if (!missing(val)) {
        if (!identical(val, private$.learner)) {
          stop("$learner is read-only.")
        }
      }
      private$.learner
    },
    learner_model = function(val) {
      if (!missing(val)) {
        if (!identical(val, private$.learner)) {
          stop("$learner is read-only.")
        }
      }
      if (is.null(self$state) || is_noop(self$state)) {
        private$.learner
      } else {
        multiplicity_recurse(self$state, clone_with_state, learner = private$.learner)
      }
    },
    predict_type = function(val) {
      if (!missing(val)) {
        assert_subset(val, names(mlr_reflections$learner_predict_types[[private$.learner$task_type]]))
        private$.learner$predict_type = val
      }
      private$.learner$predict_type
    }
  ),
  private = list(
    .learner = NULL,

    .train = function(inputs) {
      on.exit({private$.learner$state = NULL})
      task = inputs[[1L]]
      self$state = private$.learner$train(task)$state

      list(NULL)
    },

    .predict = function(inputs) {
      on.exit({private$.learner$state = NULL})
      task = inputs[[1]]
      private$.learner$state = self$state
      list(private$.learner$predict(task))
    }
  )
)

mlr_pipeops$add("learner", PipeOpLearner,
  list(R6Class("Learner",
    public = list(id = "learner", task_type = "classif", param_set = ParamSet$new()),
    active = list(encapsulate = function() c(train = "none", predict = "none")))$new()))

#' @export
as_learner.PipeOp = function(x, clone = FALSE) {
  as_learner(as_graph(x))
}
