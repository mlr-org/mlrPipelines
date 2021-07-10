ImpactEncoderClassif = R6Class("ImpactEncoderClassif", inherit = Learner,
  public = list(
    initialize = function(id, param_set = ParamSet$new(), properties = character(), data_formats = "data.table", packages = character(), man = NA_character_) {
      super$initialize(id = id, task_type = "classif", param_set = param_set, predict_types = "impact", feature_types = c("factor", "ordered"), properties = properties, packages = packages, man = man)
    }
  ),
  private = list(
    .predict = function(task) {
      impact = get_impact(task$data(cols = task$feature_names), model = self$state$model)
      list(response = factor(rep_len(NA_character_, length.out = task$nrow), levels = task$levels(task$target_names)[[1L]]), impact = impact)
    }
  )
  # FIXME: check for the structure of the model saved during train
)

ImpactEncoderRegr = R6Class("ImpactEncoderRegr", inherit = Learner,
  public = list(
    initialize = function(id, param_set = ParamSet$new(), properties = character(), data_formats = "data.table", packages = character(), man = NA_character_) {
      super$initialize(id = id, task_type = "regr", param_set = param_set, predict_types = "impact", feature_types = c("factor", "ordered"), properties = properties, packages = packages, man = man)
    }
  ),
  private = list(
    .predict = function(task) {
      impact = get_impact(task$data(cols = task$feature_names), model = self$state$model)
      list(response = rep_len(NA_real_, length.out = task$nrow), impact = impact)
    }
  )
  # FIXME: check for the structure of the model saved during train
)

ImpactEncoderClassifSimple = R6Class("ImpactEncoderClassifSimple", inherit = ImpactEncoderClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamDbl$new("smoothing", lower = 0, upper = Inf, tags = c("train", "required")),
        ParamLgl$new("impute_zero", tags = c("train", "required"))
      ))
      ps$values = list(smoothing = 1e-4, impute_zero = FALSE)
      super$initialize(
        id = "encode.impact.classif.simple",
        param_set = ps,
        properties = c("twoclass", "multiclass"),
        man = "FIXME"
      )
    }
  ),
  private = list(
    .train = function(task) {
      dt = task$data(cols = task$feature_names)
      target = task$truth()
      smoothing = self$param_set$values$smoothing
      model = sapply(dt, function(col) {
        sapply(levels(target), function(tl) {
          tprop = (sum(target == tl) + smoothing) / (length(target) + 2 * smoothing)
          tplogit = log(tprop / (1 - tprop))
          map_dbl(c(stats::setNames(levels(col), levels(col)), c(.TEMP.MISSING = NA)),
            function(cl) {
              if (!self$param_set$values$impute_zero && is.na(cl)) return(NA_real_)
              condprob = (sum(target[is.na(cl) | col == cl] == tl, na.rm = TRUE) + smoothing) / (sum(is.na(cl) | col == cl, na.rm = TRUE) + 2 * smoothing)
              cplogit = log(condprob / (1 - condprob))
              cplogit - tplogit
            }
          )
        })
      }, simplify = FALSE)
      set_class(model, "encode.impact.classif.simple_model")
    }
  )
)

ImpactEncoderRegrSimple = R6Class("ImpactEncoderRegrSimple", inherit = ImpactEncoderRegr,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamDbl$new("smoothing", lower = 0, upper = Inf, tags = c("train", "required")),
        ParamLgl$new("impute_zero", tags = c("train", "required"))
      ))
      ps$values = list(smoothing = 1e-4, impute_zero = FALSE)
      super$initialize(
        id = "encode.impact.regr.simple",
        param_set = ps,
        man = "FIXME"
      )
    }
  ),
  private = list(
    .train = function(task) {
      dt = task$data(cols = task$feature_names)
      target = task$truth()
      meanimp = mean(target)
      smoothing = self$param_set$values$smoothing
      model = sapply(dt, function(col) {
        t(t(c(sapply(levels(col), function(lvl) {
          (sum(target[col == lvl], na.rm = TRUE) + smoothing * meanimp) / (sum(col == lvl, na.rm = TRUE) + smoothing) - meanimp
        }), if (self$param_set$values$impute_zero) c(.TEMP.MISSING = 0) else c(.TEMP.MISSING = NA))))
      }, simplify = FALSE)
      set_class(model, "encode.impact.regr.simple_model")
    }
  )
)

ImpactEncoderClassifGlmm = R6Class("ImpactEncoderClassifGlmm", inherit = ImpactEncoderClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new()  # FIXME:
      super$initialize(
        id = "encode.impact.classif.glmm",
        param_set = ps,
        properties = c("twoclass", "multiclass"),
        man = "FIXME"
        # FIXME: properties missings?
      )
    }
  ),
  private = list(
    .train = function(task) {
      dt = task$data(cols = task$feature_names)
      target = task$truth()
      lvls = levels(target)
      model = if (length(lvls) > 2L) {
        # binomial glmm
        binary_target = sapply(levels(target), function(x) factor(identical(x, target)), simplify = FALSE)
        sapply(dt, function(col) {
          tmp = sapply(lvls, function(lvl) {
            fitGlmer(binary_target[[lvl]], feature = col)
          }, simplify = FALSE)
          tmp = do.call(cbind, tmp)
          colnames(tmp) = lvls
          tmp
        }, simplify = FALSE)
      } else {
        # one vs. rest binomial glmm
        sapply(dt, function(col) {
          tmp = fitGlmer(target, feature = col)
          tmp = cbind(-tmp, tmp)  # required for the other target level
          colnames(tmp) = lvls
          tmp
        }, simplify = FALSE)
      }
      set_class(model, "encode.impact.classif.glmm_model")
    }
  )
)

ImpactEncoderRegrGlmm = R6Class("ImpactEncoderRegrGlmm", inherit = ImpactEncoderRegr,
  public = list(
    initialize = function() {
      ps = ParamSet$new()  # FIXME:
      super$initialize(
        id = "encode.impact.regr.glmm",
        param_set = ps,
        man = "FIXME"
        # FIXME: properties missings?
      )
    }
  ),
  private = list(
    .train = function(task) {
      dt = task$data(cols = task$feature_names)
      target = task$truth()
      model = sapply(dt, function(col) {
        fitLmer(target, feature = col)
      }, simplify = FALSE)
      set_class(model, "encode.impact.regr.glmm_model")
    }
  )
)

get_impact = function(dt, model) {
  imap(dt, function(curdat, idx) {
    curdat = as.character(curdat)
    curdat[is.na(curdat)] = ".TEMP.MISSING"
    curdat[curdat %nin% rownames(model[[idx]])] = ".TEMP.MISSING"
    # we only want to "drop" if there are no column names
    # otherwise we want the naming scheme <original feature name>.<target level>
    model[[idx]][match(curdat, rownames(model[[idx]])), , drop = is.null(colnames(model[[idx]]))]
  })
}

# Regr helper function around lme4::lmer
# FIXME: params
fitLmer = function(target, feature) {
  args = list(formula = y ~ 1 + (1 | x),
    data = data.table(y = target, x = feature),
    na.action = na.omit,
    control = lme4::lmerControl(calc.derivs = FALSE)
  )
  mod = invoke(lme4::lmer, .args = args)
  coefs = stats::coef(mod)$x
  lvls = rownames(coefs)
  coefs = coefs[[1L]]
  names(coefs) = lvls
  intercept = unname(lme4::fixef(mod))
  coefs[is.na(coefs)] = intercept
  coefs = c(coefs, .TEMP.MISSING = intercept)
  t(t(coefs))
}

# Classif helper function around lme4::glmer
# FIXME: params
fitGlmer = function(target, feature) {
  args = list(formula = y ~ 1 + (1 | x),
    data = data.table(y = target, x = feature),
    family = stats::binomial,
    na.action = na.omit,
    control = lme4::glmerControl(calc.derivs = FALSE)
  )
  mod = invoke(lme4::glmer, .args = args)
  coefs = stats::coef(mod)$x
  lvls = rownames(coefs)
  coefs = coefs[[1L]]
  names(coefs) = lvls
  intercept = unname(lme4::fixef(mod))
  coefs[is.na(coefs)] = intercept
  coefs = c(coefs, .TEMP.MISSING = intercept)
  t(t(coefs))
}
