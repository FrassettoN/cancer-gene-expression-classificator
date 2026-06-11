source("renv/activate.R")
source("R/HMCC/features_selection.r")

# Build all folds
# Read the k-fold split file
read_kfold_txt <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = " ")
  blocks <- regmatches(txt, gregexpr("\\[[^\\]]*\\]", txt, perl = TRUE))[[1]]

  if (length(blocks) %% 2 != 0) {
    stop("Malformed split file: odd number of bracket blocks.")
  }

  parse_block <- function(b) {
    b <- gsub("^\\[|\\]$", "", b)
    scan(text = b, what = integer(), quiet = TRUE)
  }

  folds <- vector("list", length(blocks) / 2)
  j <- 1

  for (i in seq(1, length(blocks), by = 2)) {
    train_idx0 <- parse_block(blocks[i])
    test_idx0 <- parse_block(blocks[i + 1])

    folds[[j]] <- list(
      train = train_idx0 + 1L,
      test  = test_idx0 + 1L
    )
    j <- j + 1
  }

  folds
}

# Sort a fold's training set by label
order_by_label <- function(X, y, class_order = c(0, 1)) {
  y <- as.vector(y)

  if (nrow(X) != length(y)) {
    stop(sprintf(
      "Shape mismatch: X has %d samples but y has %d labels",
      nrow(X), length(y)
    ))
  }

  if (!all(y %in% class_order)) {
    bad <- unique(y[!y %in% class_order])
    stop(sprintf(
      "Unexpected labels in y: %s. Expected only %s.",
      paste(bad, collapse = ", "),
      paste(class_order, collapse = ", ")
    ))
  }

  rank <- match(y, class_order)
  idx <- order(rank, method = "radix")

  list(
    X = X[idx, , drop = FALSE],
    y = y[idx]
  )
}


CANCER_TO_GSE <- c(
  "Bladder Urothelial Carcinoma" = "GSE13507",
  "Breast invasive carcinoma cancer" = "GSE39004",
  "Breast cancer TCGA" = "TCGA-BRCA",
  "Colon adenocarcinoma" = "GSE41657",
  "Esophageal carcinoma" = "GSE20347",
  "Head and Neck squamous cell carcinoma" = "GSE6631",
  "Kidney Chromophobe" = "GSE15641_1",
  "Kidney renal clear cell carcinoma" = "GSE15641_2",
  "Kidney renal papillary cell carcinoma" = "GSE15641_3",
  "Liver hepatocellular carcinoma" = "GSE45267",
  "Lung squamous cell carcinoma" = "GSE33479",
  "Lung adenocarcinoma" = "GSE10072",
  "Prostate adenocarcinoma" = "GSE6919",
  "Rectum adenocarcinoma" = "GSE20842",
  "Stomach adenocarcinoma" = "GSE2685",
  "Thyroid carcinoma" = "GSE33630",
  "Uterine Corpus Endometrial Carcinoma" = "GSE17025"
)

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
root_out <- file.path("results", paste0("HMCC_", timestamp))
dir.create(root_out, showWarnings = FALSE, recursive = TRUE)
logits_out <- file.path(root_out, "logits")
dir.create(logits_out, showWarnings = FALSE, recursive = TRUE)
features_out <- file.path(root_out, "selected_features")
dir.create(features_out, showWarnings = FALSE, recursive = TRUE)

for (cancer in names(CANCER_TO_GSE)) {
  gse <- CANCER_TO_GSE[[cancer]]

  dataset_path <- file.path("data/processed", paste0(gse, "_trasp_mod.csv"))
  split_path <- file.path("data/splits/kfold", paste0("splits_kfold_", cancer, ".txt"))

  if (!file.exists(dataset_path)) {
    message("Skipping ", cancer, " (missing dataset): ", dataset_path)
    next
  }
  if (!file.exists(split_path)) {
    message("Skipping ", cancer, " (missing split): ", split_path)
    next
  }

  message("Processing: ", cancer, " | ", gse)

  df <- read.csv(
    dataset_path,
    header = FALSE,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  n_feat <- ncol(df) - 1
  colnames(df) <- c("label", paste0("gene_", 0:(n_feat - 1)))
  df$sample_id <- paste0("sample_", seq_len(nrow(df)))

  folds <- read_kfold_txt(split_path)

  fold_sets <- lapply(seq_along(folds), function(k) {
    tr <- folds[[k]]$train
    te <- folds[[k]]$test

    X_train <- df[tr, -1, drop = FALSE]
    y_train <- df[tr, 1, drop = TRUE]
    X_test <- df[te, -1, drop = FALSE]
    y_test <- df[te, 1, drop = TRUE]

    list(
      train = list(X = X_train, y = y_train),
      train_sorted = order_by_label(X_train, y_train, class_order = c(0, 1)),
      test = list(X = X_test, y = y_test),
      train_ids = df$sample_id[tr],
      test_ids = df$sample_id[te]
    )
  })

  fold_n_groups <- lapply(fold_sets, function(f) {
    as.integer(table(factor(f$train$y, levels = c(0, 1))))
  })

  cancer_out <- file.path(logits_out, cancer)
  dir.create(cancer_out, showWarnings = FALSE, recursive = TRUE)
  RF_out <- file.path(cancer_out, "RF")
  dir.create(RF_out, showWarnings = FALSE, recursive = TRUE)
  SVMR_out <- file.path(cancer_out, "SVMR")
  dir.create(SVMR_out, showWarnings = FALSE, recursive = TRUE)

  for (k in seq_along(fold_sets)) {
    n_groups <- fold_n_groups[[k]]
    train_x <- fold_sets[[k]]$train_sorted$X

    train_for_gs <- t(as.matrix(train_x))
    storage.mode(train_for_gs) <- "numeric"

    if (ncol(train_for_gs) != sum(n_groups)) {
      stop(paste("Fold", k, "for", gse, "has incompatible dimensions for GS."))
    }

    if (gse == "TCGA-BRCA") {
      top.df <- GS(train_for_gs, n_groups = n_groups, N_top = 30, DE_ANOVA = FALSE, DE_Ttest = FALSE)
    } else {
      top.df <- GS(train_for_gs, n_groups = n_groups, N_top = 30)
    }

    # Genes selected by GS (columns used for training)
    selected_genes <- setdiff(colnames(top.df), "DataLabels")

    selected_cancer_out = file.path(features_out, cancer)
    dir.create(selected_cancer_out, showWarnings = FALSE, recursive = TRUE)
    out_file <- file.path(selected_cancer_out, sprintf("selected_features_%d.txt", k))
    selected_ids <- ifelse(grepl("^gene_", selected_genes),
      sub("^gene_", "", selected_genes),
      selected_genes
    )
    writeLines(selected_ids, con = out_file)

    # Classification
    attach(top.df)
    x <- subset(top.df, select = -DataLabels)

    # Build test set with exactly the same feature columns/order
    X_test_full <- fold_sets[[k]]$test$X
    X_test_top <- X_test_full[, selected_genes, drop = FALSE]

    # # Random Forest probabilities (RF has no native logits)

    # Train on selected training features
    model.rf <- randomForest(DataLabels ~ ., data = top.df)

    # Predict class and probabilities on test fold
    pred_class <- predict(model.rf, newdata = X_test_top, type = "response")
    pred_prob <- predict(model.rf, newdata = X_test_top, type = "prob")

    # Optional: map original test labels (0/1) to GS labels (1/2)
    # GS creates DataLabels as class index order after sorting (0 first, then 1)
    y_test_01 <- fold_sets[[k]]$test$y
    y_test_gs <- factor(ifelse(y_test_01 == 0, 1, 2), levels = levels(top.df$DataLabels))

    # Optional quick accuracy
    acc <- mean(pred_class == y_test_gs)
    message(sprintf("Fold %02d | %s | accuracy=%.4f", k, gse, acc))

    # Save per-fold outputs (avoid overwriting same file each iteration)
    pred_out <- data.frame(
      y_true = y_test_gs,
      y_pred = pred_class
    )

    write.table(as.data.frame(pred_prob),
      file = file.path(RF_out, sprintf("logits_%s_%d_RF.txt", cancer, k)),
      row.names = FALSE, col.names = FALSE, sep = " "
    )

    # SVM-RBF (e1071)
    model.svmr <- svm(
      DataLabels ~ .,
      data = top.df,
      kernel = "radial",
      probability = TRUE
    )

    # Predict class and probabilities on test fold
    svm_pred_class <- predict(model.svmr, newdata = X_test_top, probability = TRUE)
    svm_pred_prob <- attr(svm_pred_class, "probabilities")

    svm_acc <- mean(svm_pred_class == y_test_gs)
    message(sprintf("Fold %02d | %s | SVM-r accuracy=%.4f", k, gse, svm_acc))

    write.table(as.data.frame(svm_pred_prob),
      file = file.path(SVMR_out, sprintf("logits_%s_%d_SVMR.txt", cancer, k)),
      row.names = FALSE, col.names = FALSE, sep = " "
    )
  }
}
