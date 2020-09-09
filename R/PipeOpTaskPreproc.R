#' @title PipeOpTaskPreproc
#'
#' @usage NULL
#' @format Abstract [`R6Class`] inheriting from [`PipeOp`].
#'
#' @description
#' Base class for handling most "preprocessing" operations. These
#' are operations that have exactly one [`Task`][mlr3::Task] input and one [`Task`][mlr3::Task] output,
#' and expect the column layout of these [`Task`][mlr3::Task]s during input and output
#' to be the same.
#'
#' Users must implement `private$.train_task()` and `private$.predict_task()`, which have a [`Task`][mlr3::Task]
#' input and should return that [`Task`][mlr3::Task]. The [`Task`][mlr3::Task] should, if possible, be
#' manipulated in-place, and should not be cloned.
#'
#' Alternatively, the `private$.train_dt()` and `private$.predict_dt()` functions can be implemented, which operate on
#' [`data.table`][data.table::data.table] objects instead. This should generally only be done if all
#' data is in some way altered (e.g. PCA changing all columns to principal components) and not if only
#' a few columns are added or removed (e.g. feature selection) because this should be done at the [`Task`][mlr3::Task]-level
#' with `private$.train_task()`. The `private$.select_cols()` function can be overloaded for `private$.train_dt()` and `private$.predict_dt()`
#' to operate only on subsets of the [`Task`][mlr3::Task]'s data, e.g. only on numerical columns.
#'
#' If the `can_subset_cols` argument of the constructor is `TRUE` (the default), then the hyperparameter `affect_columns`
#' is added, which can limit the columns of the [`Task`][mlr3::Task] that is modified by the [`PipeOpTaskPreproc`]
#' using a [`Selector`] function. Note this functionality is entirely independent of the `private$.select_cols()` functionality.
#'
#' [`PipeOpTaskPreproc`] is useful for operations that behave differently during training and prediction. For operations
#' that perform essentially the same operation and only need to perform extra work to build a `$state` during training,
#' the [`PipeOpTaskPreprocSimple`] class can be used instead.
#'
#' @section Construction:
#' ```
#' PipeOpTaskPreproc$new(id, param_set = ParamSet$new(), param_vals = list(), can_subset_cols = TRUE,
#'   packages = character(0), task_type = "Task", tags = NULL, feature_types = mlr_reflections$task_feature_types)
#' ```
#'
#' * `id` :: `character(1)`\cr
#'   Identifier of resulting object. See `$id` slot of [`PipeOp`].
#' * `param_set` :: [`ParamSet`][paradox::ParamSet]\cr
#'   Parameter space description. This should be created by the subclass and given to `super$initialize()`.
#' * `param_vals` :: named `list`\cr
#'   List of hyperparameter settings, overwriting the hyperparameter settings given in `param_set`. The
#'   subclass should have its own `param_vals` parameter and pass it on to `super$initialize()`. Default `list()`.
#' * `can_subset_cols` :: `logical(1)`\cr
#'   Whether the `affect_columns` parameter should be added which lets the user limit the columns that are
#'   modified by the [`PipeOpTaskPreproc`]. This should generally be `FALSE` if the operation adds or removes
#'   rows from the [`Task`][mlr3::Task], and `TRUE` otherwise. Default is `TRUE`.
#' * packages :: `character`\cr
#'   Set of all required packages for the [`PipeOp`]'s `private$.train()` and `private$.predict()` methods. See `$packages` slot.
#'   Default is `character(0)`.
#' * `task_type` :: `character(1)`\cr
#'   The class of [`Task`][mlr3::Task] that should be accepted as input and will be returned as output. This
#'   should generally be a `character(1)` identifying a type of [`Task`][mlr3::Task], e.g. `"Task"`, `"TaskClassif"` or
#'   `"TaskRegr"` (or another subclass introduced by other packages). Default is `"Task"`.
#' * tags :: `character` | `NULL`\cr
#'   Tags of the resulting `PipeOp`. This is added to the tag `"data transform"`. Default `NULL`.
#'* `feature_types` :: `character`\cr
#'   Feature types affected by the `PipeOp`. See `private$.select_cols()` for more information.
#'   Defaults to all available feature types.
#'
#' @section Input and Output Channels:
#' [`PipeOpTaskPreproc`] has one input channel named `"input"`, taking a [`Task`][mlr3::Task], or a subclass of
#' [`Task`][mlr3::Task] if the `task_type` construction argument is given as such; both during training and prediction.
#'
#' [`PipeOpTaskPreproc`] has one output channel named `"output"`, producing a [`Task`][mlr3::Task], or a subclass;
#' the [`Task`][mlr3::Task] type is the same as for input; both during training and prediction.
#'
#' The output [`Task`][mlr3::Task] is the modified input [`Task`][mlr3::Task] according to the overloaded
#' `private$.train_task()`/`private$.predict_taks()` or `private$.train_dt()`/`private$.predict_dt()` functions.
#'
#' @section State:
#' The `$state` is a named `list`; besides members added by inheriting classes, the members are:
#' * `affect_cols` :: `character`\cr
#'   Names of features being selected by the `affect_columns` parameter, if present; names of *all* present features otherwise.
#' * `intasklayout` :: [`data.table`]\cr
#'   Copy of the training [`Task`][mlr3::Task]'s `$feature_types` slot. This is used during prediction to ensure that
#'   the prediction [`Task`][mlr3::Task] has the same features, feature layout, and feature types as during training.
#' * `outtasklayout` :: [`data.table`]\cr
#'   Copy of the trained [`Task`][mlr3::Task]'s `$feature_types` slot. This is used during prediction to ensure that
#'   the [`Task`][mlr3::Task] resulting from the prediction operation has the same features, feature layout, and feature types as after training.
#' * `dt_columns` :: `character`\cr
#'   Names of features selected by the `private$.select_cols()` call during training. This is only present if the `private$.train_dt()` functionality is used,
#'   and not present if the `private$.train_task()` function is overloaded instead.
#' * `feature_types` :: `character`\cr
#'   Feature types affected by the `PipeOp`. See `private$.select_cols()` for more information.
#'
#' @section Parameters:
#' * `affect_columns` :: `function` | [`Selector`] | `NULL` \cr
#'   What columns the [`PipeOpTaskPreproc`] should operate on. This parameter is only present if the constructor is called with
#'   the `can_subset_cols` argument set to `TRUE` (the default).\cr
#'   The parameter must be a [`Selector`] function, which takes a [`Task`][mlr3::Task] as argument and returns a `character`
#'   of features to use.\cr
#'   See [`Selector`] for example functions. Defaults to `NULL`, which selects all features.
#'
#' @section Internals:
#' [`PipeOpTaskPreproc`] is an abstract class inheriting from [`PipeOp`]. It implements the `private$.train()` and
#' `$.predict()` functions. These functions perform checks and go on to call `private$.train_task()` and `private$.predict_task()`.
#' A subclass of [`PipeOpTaskPreproc`] may implement these functions, or implement `private$.train_dt()` and `private$.predict_dt()` instead.
#' This works by having the default implementations of `private$.train_task()` and `private$.predict_task()` call `private$.train_dt()` and `private$.predict_dt()`,
#' respectively.
#'
#' The `affect_columns` functionality works by unsetting columns by removing their "col_role" before
#' processing, and adding them afterwards by setting the col_role to `"feature"`.
#'
#' @section Fields:
#' Fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Methods inherited from [`PipeOp`], as well as:
#' * `.train_task`\cr
#'   ([`Task`][mlr3::Task]) -> [`Task`][mlr3::Task]\cr
#'   Called by the [`PipeOpTaskPreproc`]'s implementation of `private$.train()`. Takes a single [`Task`][mlr3::Task] as input
#'   and modifies it (ideally in-place without cloning) while storing information in the `$state` slot. Note that unlike
#'   `$.train()`, the argument is *not* a list but a singular [`Task`][mlr3::Task], and the return object is also *not* a list but
#'   a singular [`Task`][mlr3::Task]. Also, contrary to `private$.train()`, the `$state` being generated must be a `list`, which
#'   the [`PipeOpTaskPreproc`] will add additional slots to (see Section *State*). Care should be taken to avoid name collisions between
#'   `$state` elements added by `private$.train_task()` and [`PipeOpTaskPreproc`].\cr
#'   By default this function calls the `private$.train_dt()` function, but it can be overloaded to perform operations on the [`Task`][mlr3::Task]
#'   directly.
#' * `.predict_task`\cr
#'   ([`Task`][mlr3::Task]) -> [`Task`][mlr3::Task]\cr
#'   Called by the [`PipeOpTaskPreproc`]'s implementation of `$.predict()`. Takes a single [`Task`][mlr3::Task] as input
#'   and modifies it (ideally in-place without cloning) while using information in the `$state` slot. Works analogously to
#'   `private$.train_task()`. If `private$.predict_task()` should only be overloaded if `private$.train_task()` is overloaded (i.e. `private$.train_dt()` is *not* used).
#' * `.train_dt(dt, levels, target)` \cr
#'   ([`data.table`], named `list`, `any`) -> [`data.table`] | `data.frame` | `matrix` \cr
#'   Train [`PipeOpTaskPreproc`] on `dt`, transform it and store a state in `$state`. A transformed object must be returned
#'   that can be converted to a `data.table` using [`as.data.table`]. `dt` does not need to be copied deliberately, it
#'   is possible and encouraged to change it in-place.\cr
#'   The `levels` argument is a named list of factor levels for factorial or character features. The `target` argument
#'   contains the `$truth()` information of the training [`Task`][mlr3::Task]; its type depends on the [`Task`][mlr3::Task]
#'   type being trained on.\cr
#'   This method can be overloaded when inheriting from [`PipeOpTaskPreproc`], together with `private$.predict_dt()` and optionally
#'   `private$.select_cols()`; alternatively, `private$.train_task()` and `private$.predict_task()` can be overloaded.
#' * `.predict_dt(dt, levels)` \cr
#'   ([`data.table`], named `list`) -> [`data.table`] | `data.frame` | `matrix` \cr
#'   Predict on new data in `dt`, possibly using the stored `$state`. A transformed object must be returned
#'   that can be converted to a `data.table` using [`as.data.table`]. `dt` does not need to be copied deliberately, it
#'   is possible and encouraged to change it in-place.\cr
#'   The `levels` argument is a named list of factor levels for factorial or character features.\cr
#'   This method can be overloaded when inheriting `PipeOpTaskPreproc`, together with `private$.train_dt()` and optionally
#'   `private$.select_cols()`; alternatively, `private$.train_task()` and `private$.predict_task()` can be overloaded.
#' * `.select_cols(task)` \cr
#'   ([`Task`][mlr3::Task]) -> `character` \cr
#'   Selects which columns the [`PipeOp`] operates on, if `private$.train_dt()` and `private$.predict_dt()` are overloaded. This function
#'   is not called if `private$.train_task()` and `private$.predict_task()` are overloaded. In contrast to
#'   the `affect_columns` parameter. `private$.select_cols()` is for the *inheriting class* to determine which columns
#'   the operator should function on, e.g. based on feature type, while `affect_columns` is a way for the *user*
#'   to limit the columns that a [`PipeOpTaskPreproc`] should operate on.\cr
#'   This method can optionally be overloaded when inheriting [`PipeOpTaskPreproc`], together with `private$.train_dt()` and
#'   `private$.predict_dt()`; alternatively, `private$.train_task()` and `private$.predict_task()` can be overloaded.\cr
#'   If this method is not overloaded, it defaults to selecting of type indicated by the `feature_types` construction argument.
#'
#' @family mlr3pipelines backend related
#' @family PipeOps
#' @include PipeOp.R
#' @export
PipeOpTaskPreproc = R6Class("PipeOpTaskPreproc",

  inherit = PipeOp,

  public = list(
    initialize = function(id, param_set = ParamSet$new(), param_vals = list(), can_subset_cols = TRUE,
      packages = character(0), task_type = "Task", tags = NULL, feature_types = mlr_reflections$task_feature_types) {
      if (can_subset_cols) {
        acp = ParamUty$new("affect_columns", custom_check = check_function_or_null, default = selector_all(), tags = "train")
        if (inherits(param_set, "ParamSet")) {
          param_set$add(acp)
        } else {
          private$.affectcols_ps = ParamSet$new(list(acp))
          param_set = c(param_set, alist(private$.affectcols_ps))
        }
      }
      private$.feature_types = assert_subset(feature_types, mlr_reflections$task_feature_types)
      super$initialize(id = id, param_set = param_set, param_vals = param_vals,
        input = data.table(name = "input", train = task_type, predict = task_type),
        output = data.table(name = "output", train = task_type, predict = task_type),
        packages = packages, tags = c(tags, "data transform")
      )
    }
  ),
  active = list(
    feature_types = function(types) {
      if (!missing(types)) stop("feature_types can not be changed. Use the 'affect_columns' hyperparameter instead!")
      private$.feature_types
    }
  ),
  private = list(
    .affectcols_ps = NULL,
    .feature_types = NULL,

    .train = function(inputs) {
      intask = inputs[[1]]$clone(deep = TRUE)
      do_subset = !is.null(self$param_set$values$affect_columns)
      affected_cols = intask$feature_names
      if (do_subset) {
        affected_cols = self$param_set$values$affect_columns(intask)
        assert_subset(affected_cols, intask$feature_names, empty.ok = TRUE)

        remove_roles = map(intask$col_roles, .f = setdiff, y = affected_cols)
        remove_cols = setdiff(intask$feature_names, affected_cols)
        intask$col_roles = map(intask$col_roles, .f = setdiff, y = remove_cols)
      }
      intasklayout = copy(intask$feature_types)

      intask = private$.train_task(intask)

      self$state$affected_cols = affected_cols
      self$state$intasklayout = intasklayout
      self$state$outtasklayout = copy(intask$feature_types)
      self$state$outtaskshell = intask$data(rows = intask$row_ids[0])

      if (do_subset) {
        # FIXME: this fails if .train_task added a column with the same name
        keys = names(remove_roles)
        roles = mapply(union, remove_roles[keys], intask$col_roles[keys])
        names(roles) = keys
        intask$col_roles = roles
      }
      list(intask)
    },

    .predict = function(inputs) {
      intask = inputs[[1]]$clone(deep = TRUE)
      do_subset = !is.null(self$param_set$values$affect_columns)

      if (do_subset) {
        affected_cols = self$param_set$values$affect_columns(intask)

        remove_roles = map(intask$col_roles, .f = setdiff, y = affected_cols)
        remove_cols = setdiff(intask$feature_names, self$state$affected_cols)
        intask$col_roles = map(intask$col_roles, .f = setdiff, y = remove_cols)
      }
      if (!isTRUE(all.equal(self$state$intasklayout, intask$feature_types, ignore.row.order = TRUE))) {
        stopf("Input task during prediction of %s does not match input task during training.", self$id)
      }
      if (!intask$nrow) {
        # don't put the burdon of having to deal with 0-row tasks on the individual PipeOps.
        # Instead we do this here: add the correct columns (which are all empty)
        targetlayout = self$state$outtasklayout
        intask$
          select(fintersect(intask$feature_types, targetlayout)$id)$
          cbind(self$state$outtaskshell[, fsetdiff(targetlayout, intask$feature_types)$id, with = FALSE])
      } else {
        intask = private$.predict_task(intask)
      }

      if (!isTRUE(all.equal(self$state$outtasklayout, intask$feature_types, ignore.row.order = TRUE))) {
        stopf("Processed output task during prediction of %s does not match output task during training.", self$id)
      }
      if (do_subset) {
        # FIXME: see train fixme: this fails if .train_task added a column with the same name
        keys = names(remove_roles)
        roles = mapply(union, remove_roles[keys], intask$col_roles[keys])
        names(roles) = keys
        intask$col_roles = roles
      }
      list(intask)
    },
    .train_task = function(task) {
      dt_columns = private$.select_cols(task)
      cols = dt_columns
      if (!length(cols)) {
        self$state = list(dt_columns = dt_columns)
        return(task)
      }
      dt = task$data(cols = cols)
      target = task$truth()
      dt = as.data.table(private$.train_dt(dt, task$levels(cols), target))
      self$state$dt_columns = dt_columns
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },

    .predict_task = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      dt = task$data(cols = cols)
      dt = as.data.table(private$.predict_dt(dt, task$levels(cols)))
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },

    .train_dt = function(dt, levels, target) stop("Abstract."),

    .predict_dt = function(dt, levels) stop("Abstract."),

    .select_cols = function(task) selector_type(private$.feature_types)(task)

  )
)

#' @title PipeOpTaskPreprocSimple
#'
#' @usage NULL
#' @format Abstract [`R6Class`] inheriting from [`PipeOpTaskPreproc`]/[`PipeOp`].
#'
#' @description
#' Base class for handling many "preprocessing" operations
#' that perform essentially the same operation during training and prediction.
#' Instead implementing a `private$.train_task()` and a `private$.predict_task()` operation, only
#' a `private$.get_state()` and a `private$.transform()` operation needs to be defined,
#' both of which take one argument: a [`Task`][mlr3::Task].
#'
#' Alternatively, analogously to the [`PipeOpTaskPreproc`] approach of offering `private$.train_dt()`/`private$.predict_dt()`,
#' the `private$.get_state_dt()` and `private$.transform_dt()` functions may be implemented.
#'
#' `private$.get_state` must not change its input value in-place and must return
#' something that will be written into `$state`
#' (which must not be NULL), `private$.transform()` should modify its argument in-place;
#' it is called both during training and prediction.
#'
#' This inherits from [`PipeOpTaskPreproc`] and behaves essentially the same.
#'
#' @section Construction:
#' ```
#' PipeOpTaskPreprocSimple$new(id, param_set = ParamSet$new(), param_vals = list(), can_subset_cols = TRUE, packages = character(0), task_type = "Task")
#' ```
#' (Construction is identical to [`PipeOpTaskPreproc`].)
#'
#' * `id` :: `character(1)`\cr
#'   Identifier of resulting object. See `$id` slot of [`PipeOp`].
#' * `param_set` :: [`ParamSet`][paradox::ParamSet]\cr
#'   Parameter space description. This should be created by the subclass and given to `super$initialize()`.
#' * `param_vals` :: named `list`\cr
#'   List of hyperparameter settings, overwriting the hyperparameter settings given in `param_set`. The
#'   subclass should have its own `param_vals` parameter and pass it on to `super$initialize()`. Default `list()`.
#' * `can_subset_cols` :: `logical(1)`\cr
#'   Whether the `affect_columns` parameter should be added which lets the user limit the columns that are
#'   modified by the [`PipeOpTaskPreprocSimple`]. This should generally be `FALSE` if the operation adds or removes
#'   rows from the [`Task`][mlr3::Task], and `TRUE` otherwise. Default is `TRUE`.
#' * packages :: `character`\cr
#'   Set of all required packages for the [`PipeOp`]'s `private$.train()` and `private$.predict()` methods. See `$packages` slot.
#'   Default is `character(0)`.
#' * `task_type` :: `character(1)`\cr
#'   The class of [`Task`][mlr3::Task] that should be accepted as input and will be returned as output. This
#'   should generally be a `character(1)` identifying a type of [`Task`][mlr3::Task], e.g. `"Task"`, `"TaskClassif"` or
#'   `"TaskRegr"` (or another subclass introduced by other packages). Default is `"Task"`.
#'
#' @section Input and Output Channels:
#' Input and output channels are inherited from [`PipeOpTaskPreproc`].
#'
#' The output during training and prediction is the [`Task`][mlr3::Task], modified by `private$.transform()` or `private$.transform_dt()`.
#'
#' @section State:
#' The `$state` is a named `list` with the `$state` elements inherited from [`PipeOpTaskPreproc`].
#'
#' @section Parameters:
#' The parameters are the parameters inherited from [`PipeOpTaskPreproc`].
#'
#' @section Internals:
#' [`PipeOpTaskPreprocSimple`] is an abstract class inheriting from [`PipeOpTaskPreproc`] and implementing the
#' `private$.train_task()` and `private$.predict_task()` functions. A subclass of [`PipeOpTaskPreprocSimple`] may implement the
#' functions `private$.get_state()` and `private$.transform()`, or alternatively the functions `private$.get_state_dt()` and `private$.transform_dt()`
#' (as well as `private$.select_cols()`, in the latter case). This works by having the default implementations of
#' `private$.get_state()` and `private$.transform()` call `private$.get_state_dt()` and `private$.transform_dt()`.
#'
#' @section Fields:
#' Fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Methods inherited from [`PipeOpTaskPreproc`], as well as:
#' * `.get_state(task)` \cr
#'   ([`Task`][mlr3::Task]) -> named `list`\cr
#'   Store create something that will be stored in `$state` during training phase of `PipeOpTaskPreprocSimple`.
#'   The state can then influence the `private$.transform()` function. Note that `private$.get_state()` must *return* the state, and
#'   should not store it in `$state`. It is not strictly necessary to implement either `private$.get_state()` or `private$.get_state_dt()`;
#'   if they are not implemented, the state will be stored as `list()`. \cr
#'   This method can optionally be overloaded when inheriting from [`PipeOpTaskPreprocSimple`], together with `private$.transform()`;
#'   alternatively, `private$.get_state_dt()` (optional) and `private$.transform_dt()` (and possibly `private$.select_cols()`, from [`PipeOpTaskPreproc`])
#'   can be overloaded.
#' * `.transform(task)` \cr
#'   ([`Task`][mlr3::Task]) -> [`Task`][mlr3::Task]\cr
#'   Predict on new data in `task`, possibly using the stored `$state`. `task` should not be cloned, instead it should be
#'   changed in-place. This method is called both during training and prediction phase, and should essentially behave the
#'   same independently of phase. (If this is incongruent with the functionality to be implemented, then it should inherit from
#'   [`PipeOpTaskPreproc`], not from [`PipeOpTaskPreprocSimple`].) \cr
#'   This method can be overloaded when inheriting from [`PipeOpTaskPreprocSimple`], optionally with `private$.get_state()`;
#'   alternatively, `private$.get_state_dt()` (optional) and `private$.transform_dt()` (and possibly `private$.select_cols()`, from [`PipeOpTaskPreproc`])
#'   can be overloaded.
#' * `.get_state_dt(dt)` \cr
#'   ([`data.table`]) -> named `list`\cr
#'   Create something that will be stored in `$state` during training phase of `PipeOpTaskPreprocSimple`.
#'   The state can then influence the `private$.transform_dt()` function. Note that `private$.get_state_dt()` must *return* the state, and
#'   should not store it in `$state`. If neither `private$.get_state()` nor `private$.get_state_dt()` are overloaded, the state will
#'   be stored as `list()`. \cr
#'   This method can optionally be overloaded when inheriting from [`PipeOpTaskPreprocSimple`], together with `private$.transform_dt()`
#'   (and optionally `private$.select_cols()`, from [`PipeOpTaskPreproc`]); Alternatively, `private$.get_state()` (optional) and `private$.transform()`
#'   can be overloaded.
#' * `.transform_dt(dt)` \cr
#'   ([`data.table`]) -> [`data.table`] | `data.frame` | `matrix` \cr
#'   Predict on new data in `dt`, possibly using the stored `$state`. A transformed object must be returned
#'   that can be converted to a `data.table` using [`as.data.table`]. `dt` does not need to be copied deliberately, it
#'   is possible and encouraged to change it in-place. This method is called both during training and prediction phase,
#'   and should essentially behave the same independently of phase.
#'   (If this is incongruent with the functionality to be implemented, then it should inherit from
#'   [`PipeOpTaskPreproc`], not from [`PipeOpTaskPreprocSimple`].) \cr
#'   This method can optionally be overloaded when inheriting from [`PipeOpTaskPreprocSimple`], together with `private$.transform_dt()`
#'   (and optionally `private$.select_cols()`, from [`PipeOpTaskPreproc`]); Alternatively, `private$.get_state()` (optional) and `private$.transform()`
#'   can be overloaded.
#'
#' @family PipeOps
#' @family mlr3pipelines backend related
#' @export
PipeOpTaskPreprocSimple = R6Class("PipeOpTaskPreprocSimple",

  inherit = PipeOpTaskPreproc,

  private = list(
    .train_task = function(task) {
      self$state = private$.get_state(task)
      private$.transform(task)
    },
    .predict_task = function(task) private$.transform(task),

    .get_state = function(task) {
      dt_columns = private$.select_cols(task)
      cols = dt_columns
      if (!length(cols)) {
        return(list(dt_columns = dt_columns))
      }
      dt = task$data(cols = cols)
      target = task$truth()
      c(private$.get_state_dt(dt, task$levels(cols), target), list(dt_columns = dt_columns))
    },

    .transform = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      dt = task$data(cols = cols)
      dt = as.data.table(private$.transform_dt(dt, task$levels(cols)))
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },

    .get_state_dt = function(dt, levels, target) list(),

    .transform_dt = function(dt, levels) stop("Abstract")
  )
)
