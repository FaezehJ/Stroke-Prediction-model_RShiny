library(shiny)
library(shinythemes)
library(DT)
library(plotly)

library(tidymodels)
library(themis)
library(janitor)
library(vip)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)

theme_set(theme_minimal())
options(tidymodels.dark = FALSE)
set.seed(42)

#-------------------------- UI --------------------------

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Stroke Prediction — Train & Evaluate"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      fileInput("file", "Upload CSV", accept = c(".csv")),
      checkboxInput("header", "Header", TRUE),
      textInput("target", "Target column", value = "stroke"),
      
      helpText("Expected columns include: gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status, stroke"),
      hr(),
      radioButtons("model_type", "Model", choices = c("Random Forest" = "rf",
                                                      "Lasso Logistic" = "logit")),
      numericInput("trees", "RF trees", value = 800, min = 100, step = 100),
      sliderInput("class_wt", "RF class weight for 'yes' (no=1, yes= ?)", min = 1, max = 30, value = 10),
      checkboxInput("use_smote", "Use SMOTE (train only)", TRUE),
      hr(),
      sliderInput("split", "Train proportion", min = 0.6, max = 0.9, value = 0.8, step = 0.05),
      numericInput("folds", "CV folds", value = 5, min = 3, max = 10),
      actionButton("train", "Train", class = "btn btn-primary"),
      hr(),
      sliderInput("thr", "Decision threshold", min = 0.01, max = 0.5, value = 0.25, step = 0.01),
      helpText("Lower threshold → higher sensitivity (recall), lower specificity.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data",
                 br(),
                 fluidRow(
                   column(6, h4("Class balance (full data)"), verbatimTextOutput("class_balance")),
                   column(6, h4("Train/Test sizes"), verbatimTextOutput("split_sizes"))
                 ),
                 DTOutput("preview")
        ),
        tabPanel("Metrics",
                 br(),
                 fluidRow(
                   column(6, h4("At 0.5 threshold"), tableOutput("metrics_default")),
                   column(6, h4("At chosen threshold"), tableOutput("metrics_chosen"))
                 ),
                 fluidRow(
                   column(6, h4("Confusion matrix (chosen threshold)"), tableOutput("cmatrix")),
                   column(6, h4("Chosen threshold"), verbatimTextOutput("thr_shown"))
                 )
        ),
        tabPanel("Curves",
                 br(),
                 fluidRow(
                   column(6, h4("ROC Curve"), plotlyOutput("roc_plot")),
                   column(6, h4("Precision–Recall Curve"), plotlyOutput("pr_plot"))
                 )
        ),
        tabPanel("Importance (RF)",
                 br(),
                 helpText("Shown only for Random Forest"),
                 plotOutput("vip_plot", height = "420px")
        ),
        tabPanel("Individual",
                 br(),
                 fluidRow(
                   column(4,
                          h4("Enter individual information"),
                          numericInput("one_age", "Age (years)", value = 60, min = 0, max = 120, step = 1),
                          numericInput("one_bmi", "BMI", value = 28, min = 10, max = 60, step = 0.1),
                          numericInput("one_glucose", "Avg. glucose level (mg/dL)", value = 95, min = 50, max = 300, step = 0.1),
                          selectInput("one_gender", "Gender", choices = c("Male","Female")),
                          selectInput("one_ever_married", "Ever married", choices = c("Yes","No")),
                          selectInput("one_work_type", "Work type", choices = c("Private","Self-employed","Govt_job","children","Never_worked")),
                          selectInput("one_residence_type", "Residence type", choices = c("Urban","Rural")),
                          selectInput("one_smoking_status", "Smoking status", choices = c("never smoked","formerly smoked","smokes","Unknown")),
                          checkboxInput("one_hyp", "Hypertension", FALSE),
                          checkboxInput("one_hd",  "Heart disease", FALSE),
                          hr(),
                          actionButton("predict_one", "Predict", class = "btn btn-success")
                   ),
                   column(8,
                          h4("Result"),
                          tags$div(style="font-size:18px;margin-bottom:10px;",
                                   "Using the model trained in the “Train” tab and the current threshold slider."),
                          fluidRow(
                            column(6,
                                   wellPanel(
                                     h5("Predicted probability of stroke (yes)"),
                                     htmlOutput("one_prob_html")
                                   )
                            ),
                            column(6,
                                   wellPanel(
                                     h5("Predicted class at current threshold"),
                                     htmlOutput("one_class_html")
                                   )
                            )
                          ),
                          br(),
                          h4("What changed the prediction?"),
                          plotOutput("one_vip_local", height = "320px")
                   )
                 )
        ),
        tabPanel("Predictions",
                 br(),
                 downloadButton("dl_preds", "Download predictions CSV"),
                 DTOutput("pred_table")
        )
      )
    )
  )
)


#-------------------------- SERVER --------------------------

server <- function(input, output, session) {
  
  
  
  # ---------- helpers ----------
  # clean names, coerce bmi, factorize some columns
  clean_df <- function(df) {
    names(df) <- names(df) |>
      tolower() |>
      gsub("[^a-z0-9]+", "_", x = _) |>
      sub("^_", "", x = _) |>
      sub("_$", "", x = _)
    if ("bmi" %in% names(df)) df$bmi <- suppressWarnings(as.numeric(df$bmi))
    fac_cols <- c("gender","ever_married","work_type","residence_type","smoking_status")
    for (nm in intersect(fac_cols, names(df))) df[[nm]] <- factor(df[[nm]])
    df
  }
  
  # class metrics at a given threshold (vector APIs avoid NSE headaches)
  metrics_at <- function(pred_df, tgt, thr) {
    est <- factor(ifelse(pred_df$.pred_yes >= thr, "yes", "no"),
                  levels = c("no","yes"))
    tibble::tibble(
      accuracy    = suppressWarnings(yardstick::accuracy_vec(pred_df[[tgt]], est)),
      sensitivity = suppressWarnings(yardstick::sens_vec(pred_df[[tgt]], est, event_level = "second")),
      specificity = suppressWarnings(yardstick::spec_vec(pred_df[[tgt]], est, event_level = "second")),
      ppv         = suppressWarnings(yardstick::precision_vec(pred_df[[tgt]], est, event_level = "second")),
      npv         = suppressWarnings(yardstick::npv_vec(pred_df[[tgt]], est, event_level = "second"))
    )
  }
  
  
  
  # ---------- data ----------
  raw_data <- reactive({
    req(input$file)
    read.csv(input$file$datapath, stringsAsFactors = FALSE) |>
      clean_df() |>
      dplyr::select(-dplyr::any_of("id"))   # <- removes id if present
  })
  
  
  # make target a factor (no/yes) 
  data_coerced <- reactive({
    df <- raw_data()
    tgt <- input$target
    req(tgt %in% names(df))
    
    s <- df[[tgt]]
    if (is.factor(s)) s <- as.character(s)
    if (is.character(s)) {
      s_num <- suppressWarnings(as.numeric(s))
      if (!all(s %in% c("0","1")) && any(is.na(s_num))) {
        s_num <- ifelse(tolower(s) %in% "yes", 1,
                        ifelse(tolower(s) %in% "no", 0, NA_real_))
      }
    } else if (is.numeric(s)) {
      s_num <- s
    } else {
      validate(need(FALSE, "Unexpected target type"))
    }
    
    keep <- which(s_num %in% c(0,1))
    validate(need(length(keep) > 1, "Target must contain 0/1 or yes/no"))
    df <- df[keep, , drop = FALSE]
    df[[tgt]] <- factor(s_num[keep], levels = c(0,1), labels = c("no","yes"))
    df
  })
  
  
  
  # ---------- Data tab outputs ----------
  output$preview <- DT::renderDT({
    DT::datatable(head(data_coerced(), 20),
                  options = list(scrollX = TRUE, pageLength = 20))
  })
  
  output$class_balance <- renderPrint({
    tbl <- table(data_coerced()[[input$target]])
    print(tbl)
    if ("yes" %in% names(tbl))
      cat(sprintf("\nPositives = %.2f%%\n", 100 * tbl["yes"] / sum(tbl)))
  })
  
  split_obj <- reactive({
    rsample::initial_split(data_coerced(), prop = input$split,
                           strata = dplyr::all_of(input$target))
  })
  train_dat <- reactive(rsample::training(split_obj()))
  test_dat  <- reactive(rsample::testing(split_obj()))
  
  output$split_sizes <- renderPrint({
    tr <- train_dat(); te <- test_dat()
    cat("Train:", nrow(tr), "rows   Test:", nrow(te), "rows\n")
    cat("Train class counts:\n"); print(table(tr[[input$target]]))
    cat("Test class counts:\n");  print(table(te[[input$target]]))
  })
  
  
  # ---------- recipe ----------
  rec <- reactive({
    tr  <- train_dat()
    tgt <- input$target
    
    preds <- setdiff(names(tr), c(tgt, "id"))   # ← exclude id here
    r <- recipes::recipe(stats::reformulate(preds, response = tgt), data = tr)
    
    r %>%
      recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
      recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01, other = "other") %>%
      recipes::step_dummy(recipes::all_nominal_predictors()) %>%
      recipes::step_zv(recipes::all_predictors()) %>%
      recipes::step_normalize(recipes::all_numeric_predictors()) %>%
      { if (isTRUE(input$use_smote)) themis::step_smote(., recipes::all_outcomes()) else . }
  })
  
  
  
  # ---------- model specs (fast, no tuning) ----------
  rf_spec <- reactive({
    parsnip::rand_forest(
      mtry   = NULL,               # let ranger default
      min_n  = 5,
      trees  = input$trees
    ) %>%
      parsnip::set_engine("ranger",
                          importance = "impurity",
                          class.weights = c("no" = 1, "yes" = max(1, as.numeric(input$class_wt)))) %>%
      parsnip::set_mode("classification")
  })
  
  log_spec <- reactive({
    parsnip::logistic_reg(penalty = 0.01, mixture = 1) %>%
      parsnip::set_engine("glmnet") %>%
      parsnip::set_mode("classification")
  })
  
  
  # ---------- train on click ----------
  train_results <- eventReactive(input$train, {
    tr <- train_dat(); te <- test_dat(); r <- rec()
    spec <- if (identical(input$model_type, "rf")) rf_spec() else log_spec()
    wf   <- workflows::workflow() %>% workflows::add_model(spec) %>% workflows::add_recipe(r)
    fit  <- parsnip::fit(wf, tr)
    
    probs <- predict(fit, te, type = "prob") %>%
      dplyr::bind_cols(te %>% dplyr::select(dplyr::all_of(input$target)))
    
    list(final = fit, probs = probs, train = tr, test = te)
  }, ignoreInit = TRUE)
  
  
  # --- sync choices in the Individual tab to the trained data ---
  observeEvent(train_results(), ignoreInit = TRUE, {
    tr <- train_results()$train
    
    upd <- function(id, x) updateSelectInput(session, id, choices = levels(x))
    if (is.factor(tr$gender))          upd("one_gender",          tr$gender)
    if (is.factor(tr$ever_married))    upd("one_ever_married",    tr$ever_married)
    if (is.factor(tr$work_type))       upd("one_work_type",       tr$work_type)
    if (is.factor(tr$residence_type))  upd("one_residence_type",  tr$residence_type)
    if (is.factor(tr$smoking_status))  upd("one_smoking_status",  tr$smoking_status)
  })
  
  # --- build one-row data frame from inputs (no id column) ---
  one_row <- reactive({
    req(train_results())  # need the train to get factor levels
    tr <- train_results()$train
    
    tibble::tibble(
      gender            = factor(input$one_gender,         levels = if (is.factor(tr$gender)) levels(tr$gender) else NULL),
      age               = as.numeric(input$one_age),
      hypertension      = as.integer(isTRUE(input$one_hyp)),
      heart_disease     = as.integer(isTRUE(input$one_hd)),
      ever_married      = factor(input$one_ever_married,   levels = if (is.factor(tr$ever_married)) levels(tr$ever_married) else NULL),
      work_type         = factor(input$one_work_type,      levels = if (is.factor(tr$work_type)) levels(tr$work_type) else NULL),
      residence_type    = factor(input$one_residence_type, levels = if (is.factor(tr$residence_type)) levels(tr$residence_type) else NULL),
      avg_glucose_level = as.numeric(input$one_glucose),
      bmi               = as.numeric(input$one_bmi),
      smoking_status    = factor(input$one_smoking_status, levels = if (is.factor(tr$smoking_status)) levels(tr$smoking_status) else NULL)
    )
  })
  
  # --- predict when 'Predict' is clicked ---
  one_pred <- eventReactive(input$predict_one, {
    o   <- train_results(); req(o)
    row <- one_row()
    
    # predict with the trained workflow; recipe is applied inside
    pr  <- predict(o$final, new_data = row, type = "prob")$.pred_yes
    cls <- ifelse(pr >= input$thr, "yes", "no")
    
    list(prob = as.numeric(pr), class = cls, row = row)
  }, ignoreInit = TRUE)
  
  output$one_prob_html  <- renderUI({ req(one_pred()); HTML(sprintf("<b>%.3f</b>", one_pred()$prob)) })
  output$one_class_html <- renderUI({ req(one_pred()); HTML(sprintf("<b>%s</b>", toupper(one_pred()$class))) })
  
  # --- simple local explanation: leave-one-feature-out delta vs reference ---
  output$one_vip_local <- renderPlot({
    req(one_pred())
    o      <- train_results()
    row    <- one_pred()$row
    
    # build a reference row: medians for numeric, most frequent level for factors
    tr <- o$train
    ref <- lapply(tr[names(row)], function(x) {
      if (is.numeric(x)) stats::median(x, na.rm = TRUE)
      else if (is.factor(x)) names(sort(table(x), decreasing = TRUE))[1]
      else x[1]
    }) |> as.data.frame()
    # coerce ref factor levels to match train
    for (nm in names(ref)) if (is.factor(tr[[nm]])) ref[[nm]] <- factor(ref[[nm]], levels = levels(tr[[nm]]))
    
    p_all <- predict(o$final, new_data = row, type = "prob")$.pred_yes
    
    # feature-wise deltas (replace each feature with reference value)
    deltas <- sapply(names(row), function(nm) {
      tmp <- row
      tmp[[nm]] <- ref[[nm]]
      p_tmp <- predict(o$final, new_data = tmp, type = "prob")$.pred_yes
      as.numeric(p_all - p_tmp)
    })
    
    df <- tibble::tibble(feature = names(deltas),
                         contribution = as.numeric(deltas)) |>
      dplyr::mutate(sign = ifelse(contribution >= 0, "push ↑", "push ↓")) |>
      dplyr::arrange(desc(abs(contribution))) |>
      dplyr::slice_head(n = 10)
    
    ggplot2::ggplot(df, ggplot2::aes(x = reorder(feature, contribution), y = contribution, fill = sign)) +
      ggplot2::geom_col() +
      ggplot2::coord_flip() +
      ggplot2::labs(x = NULL, y = "Δ prob(yes) vs reference", fill = NULL) +
      ggplot2::theme_minimal()
  })
  
  
  # ---------- Metrics tab ----------
  output$metrics_default <- renderTable({
    req(train_results())
    pred <- train_results()$probs
    
    # probability metrics (treat "yes" as the event)
    auc  <- yardstick::roc_auc(pred,
                               truth = !!rlang::sym(input$target),
                               .pred_yes,
                               event_level = "second")
    prau <- yardstick::pr_auc(pred,
                              truth = !!rlang::sym(input$target),
                              .pred_yes,
                              event_level = "second")
    
    # class metrics at 0.50
    class50 <- metrics_at(pred, input$target, 0.5)
    
    dplyr::bind_rows(
      tibble::tibble(metric = "roc_auc",      value = auc$.estimate),
      tibble::tibble(metric = "pr_auc",       value = prau$.estimate),
      tibble::tibble(metric = "accuracy@0.50", value = class50$accuracy),
      tibble::tibble(metric = "sens@0.50",     value = class50$sensitivity),
      tibble::tibble(metric = "spec@0.50",     value = class50$specificity),
      tibble::tibble(metric = "ppv@0.50",      value = class50$ppv),
      tibble::tibble(metric = "npv@0.50",      value = class50$npv)
    ) |>
      dplyr::mutate(value = round(value, 3))
  })
  
  output$metrics_chosen <- renderTable({
    req(train_results())
    metrics_at(train_results()$probs, input$target, input$thr) |>
      dplyr::mutate(dplyr::across(everything(), ~ round(.x, 3)))
  })
  
  output$cmatrix <- renderTable({
    req(train_results())
    pred <- train_results()$probs
    est  <- factor(ifelse(pred$.pred_yes >= input$thr, "yes", "no"),
                   levels = c("no","yes"))
    df   <- tibble::tibble(truth = pred[[input$target]], estimate = est)
    as.data.frame(yardstick::conf_mat(df, truth = truth, estimate = estimate)$table)
  })
  
  output$thr_shown <- renderPrint({
    cat("Threshold =", input$thr, "\n")
  })
  
  
  # ---------- Curves ----------
  output$roc_plot <- plotly::renderPlotly({
    req(train_results())
    pred <- train_results()$probs
    roc_df <- yardstick::roc_curve(pred, truth = !!rlang::sym(input$target), .pred_yes)
    g <- ggplot2::ggplot(roc_df, ggplot2::aes(1 - specificity, sensitivity)) +
      ggplot2::geom_line() + ggplot2::geom_abline(lty = 2) +
      ggplot2::labs(x = "1 - Specificity", y = "Sensitivity")
    plotly::ggplotly(g)
  })
  
  output$pr_plot <- plotly::renderPlotly({
    req(train_results())
    pred <- train_results()$probs
    pr_df <- yardstick::pr_curve(pred, truth = !!rlang::sym(input$target), .pred_yes)
    g <- ggplot2::ggplot(pr_df, ggplot2::aes(recall, precision)) +
      ggplot2::geom_line() + ggplot2::labs(x = "Recall", y = "Precision")
    plotly::ggplotly(g)
  })
  
  
  
  # ---------- Importance (RF) ----------
  
  output$vip_plot <- renderPlot({
    req(train_results())
    validate(need(identical(input$model_type, "rf"),
                  "Feature importance is available only for Random Forest"))
    
    # Get the parsnip fit from the workflow, then the underlying engine object
    wf_fit   <- workflows::extract_fit_parsnip(train_results()$final)
    rf_engine <- parsnip::extract_fit_engine(wf_fit)
    
    validate(need(!is.null(rf_engine$variable.importance),
                  "This RF engine did not return variable importance."))
    
    imp <- tibble::tibble(
      feature    = names(rf_engine$variable.importance),
      importance = as.numeric(rf_engine$variable.importance)
    ) |>
      dplyr::arrange(dplyr::desc(importance)) |>
      dplyr::slice_head(n = 12)
    
    ggplot2::ggplot(imp, ggplot2::aes(x = reorder(feature, importance), y = importance)) +
      ggplot2::geom_col() +
      ggplot2::coord_flip() +
      ggplot2::labs(x = NULL, y = "Importance")
  })
  
  
  
  # ---------- Predictions ----------
  pred_with_labels <- reactive({
    req(train_results())
    df <- train_results()$test
    pr <- train_results()$probs
    est <- factor(ifelse(pr$.pred_yes >= input$thr, "yes", "no"), levels = c("no","yes"))
    tibble::tibble(
      .pred_yes = pr$.pred_yes,
      .pred_no  = pr$.pred_no,
      .pred_class = est,
      truth = pr[[input$target]]
    ) %>% dplyr::bind_cols(df %>% dplyr::select(where(function(x) !is.list(x))))
  })
  
  output$pred_table <- DT::renderDT({
    DT::datatable(pred_with_labels(), options = list(scrollX = TRUE, pageLength = 15))
  })
  
  output$dl_preds <- downloadHandler(
    filename = function() paste0("predictions_", Sys.Date(), ".csv"),
    content  = function(file) readr::write_csv(pred_with_labels(), file)
  )
}


shinyApp(ui, server)

