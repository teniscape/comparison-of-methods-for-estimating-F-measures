# Installing Packages 
# install.packages("randomForest")

# Loading package 
library(class)
library(randomForest)
library(caret)

split_data <-
  function(Data, group){
    sampleframe <- rep(1:group, ceiling(nrow(Data)/group)) 
    Data$grp <- 0
    Data[,"grp"] <- sample(sampleframe, size=nrow(Data), replace=FALSE)
    
    return(Data)
  }

# Compute f values and variances
f_value_var <-
  function(a, b, train_data, test_data, num, nsample, group, sd) {
    set.seed(sd)
    nremove <- nremove_j <- nremove_w <- 0
    F_j <- F_l <- F_jl <- f_j <- f_l <- rep(0, num)
    var_j <- var_l <- cov_F_jl <- corr_jl <- var_jl <- z_score_jl <- var_j_wong <- var_l_wong <- var_jl_wong <- z_score_wong <- rep(0, num)
    X_train <- train_data[, -ncol(train_data)]
    y_train <- train_data$Rings
    idx <- vector()
    precision_j <- recall_j <- precision_l <- recall_l <- rep(0, group)
    tp_j <- fn_j <- fp_j <- tp_l <- fn_l <- fp_l <- rep(0, group)
    
    rf_cl <- randomForest(Rings ~., data = train_data, ntree = 100, proximity=T) 
    
    for (i in 1: num){
      index <- sample(1:nrow(test_data), nsample, replace=TRUE)
      test_sub <- test_data[index, ]
      X_test <- test_sub[, -ncol(test_sub)]
      y_test <- test_sub$Rings
      # y_test <- as.factor(y_test)
      
      test_j <- test_l <- test_sub
      test_j <- split_data(test_j, group = group)
      test_l <- split_data(test_l, group = group)
      
      # 1-NN model training and prediction
      predicted_j <- knn(train = X_train, 
                         test  = X_test, 
                         cl    = y_train, 
                         k     = 1)
      
      predicted_l <- predict(rf_cl, newdata = X_test) 
      
      predicted_j <- as.numeric(as.character(predicted_j))
      predicted_l <- as.numeric(as.character(predicted_l))
      y_test = as.numeric(as.character(y_test))
      
      ZA_j = predicted_j*y_test
      EZA_j = mean(ZA_j)
      A_j_Z = predicted_j*(1-y_test)
      EA_j_Z = mean(A_j_Z)
      Z_A_j = (1-predicted_j)*y_test
      EZ_A_j = mean(Z_A_j)
      F_j[i] = EZA_j/(EZA_j + a*EA_j_Z + b*EZ_A_j)
      tau_coe_squ_j = EZA_j/(EZA_j + a^2*EA_j_Z + b^2*EZ_A_j)
      var_j[i] = ((1/tau_coe_squ_j-1 + (1/F_j[i]-1)^2)/(EZA_j/F_j[i]^4))/length(y_test)
      kappa_j = 1/F_j[i]-1
      
      ZA_l = predicted_l*y_test
      EZA_l = mean(ZA_l)
      A_l_Z = predicted_l*(1-y_test)
      EA_l_Z = mean(A_l_Z)
      Z_A_l = (1-predicted_l)*y_test
      EZ_A_l = mean(Z_A_l)
      F_l[i] = EZA_l/(EZA_l + a*EA_l_Z + b*EZ_A_l)
      tau_coe_squ_l = EZA_l/(EZA_l + a^2*EA_l_Z + b^2*EZ_A_l)
      var_l[i] = ((1/tau_coe_squ_l-1 + (1/F_l[i]-1)^2)/(EZA_l/F_l[i]^4))/length(y_test)
      kappa_l = 1/F_l[i]-1
      
      vec_jj = a*A_j_Z + b*Z_A_j
      vec_ll = a*A_l_Z + b*Z_A_l
      cov_kappa_jl = 1/(length(y_test)*EZA_j*EZA_l)*(cov(vec_jj, vec_ll) + kappa_j*kappa_l*cov(ZA_j, ZA_l) - kappa_j*cov(ZA_j, vec_ll) - kappa_l*cov(ZA_l, vec_jj))
      cov_F_jl[i] = F_j[i]^2*F_l[i]^2*cov_kappa_jl

      F_jl[i] <- F_j[i] - F_l[i]
      corr_jl[i] <- cov_F_jl[i]/(sqrt(var_j[i])*sqrt(var_l[i]))
      var_jl[i] <- var_j[i]+var_l[i]-2*cov_F_jl[i]
      z_score_jl[i] <- F_jl[i]/sqrt(var_jl[i])
      
      for (k in 1: group){
        test_data_j <- test_j[test_j$grp %in% k,][,-ncol(test_j)]
        test_data_l <- test_l[test_l$grp %in% k,][,-ncol(test_l)]
        
        X_test_j <- test_data_j[,-ncol(test_data_j)]
        y_test_j <- test_data_j$Rings                 
        X_test_l <- test_data_l[,-ncol(test_data_l)]
        y_test_l <- test_data_l$Rings                 
        
        # 1-NN model training and prediction
        predicted_j <- knn(train = X_train, 
                           test  = X_test_j, 
                           cl    = y_train, 
                           k     = 1)
        predicted_j <- as.factor(predicted_j)
        
        predicted_l <- predict(rf_cl, newdata = X_test_l) 
        predicted_l <- as.factor(as.character(predicted_l))
        
        precision_j[k] <- posPredValue(predicted_j, y_test_j, positive="1")
        recall_j[k] <- sensitivity(predicted_j, y_test_j, positive="1")
        precision_l[k] <- posPredValue(predicted_l, y_test_l, positive="1")
        recall_l[k] <- sensitivity(predicted_l, y_test_l, positive="1")
        
        if (is.na(precision_j[k])){precision_j[k] <- 0}
        if (is.na(recall_j[k])){recall_j[k] <- 0}
        if (is.na(precision_l[k])){precision_l[k] <- 0}
        if (is.na(recall_l[k])){recall_l[k] <- 0}
        
        conf_j <- table(predicted_j, y_test_j)
        conf_l <- table(predicted_l, y_test_l)
        fp_j[k] <- conf_j[2]
        fn_j[k] <- conf_j[3]
        tp_j[k] <- conf_j[4]
        fp_l[k] <- conf_l[2]
        fn_l[k] <- conf_l[3]
        tp_l[k] <- conf_l[4]
        
        if (is.na(fp_j[k])){fp_j[k] <- 0}
        if (is.na(fn_j[k])){fn_j[k] <- 0}
        if (is.na(tp_j[k])){tp_j[k] <- 0}
        if (is.na(fp_l[k])){fp_l[k] <- 0}
        if (is.na(fn_l[k])){fn_l[k] <- 0}
        if (is.na(tp_l[k])){tp_l[k] <- 0}
      }
      sfp_j <- sum(fp_j)
      sfn_j <- sum(fn_j)
      stp_j <- sum(tp_j)
      sfp_l <- sum(fp_l)
      sfn_l <- sum(fn_l)
      stp_l <- sum(tp_l)
      r_j_bar <- stp_j/(stp_j+sfn_j)
      q_j_bar <- stp_j/(stp_j+sfp_j)
      r_l_bar <- stp_l/(stp_l+sfn_l)
      q_l_bar <- stp_l/(stp_l+sfp_l)
      f_j[i] <- 2*r_j_bar*q_j_bar/(r_j_bar+q_j_bar)
      f_l[i] <- 2*r_l_bar*q_l_bar/(r_l_bar+q_l_bar)
      
      w_j <- (f_j[i]-q_j_bar)/(r_j_bar-q_j_bar)
      w_l <- (f_l[i]-q_l_bar)/(r_l_bar-q_l_bar)
      var_rj <- r_j_bar*(1-r_j_bar)/(stp_j+sfn_j)
      var_qj <- q_j_bar*(1-q_j_bar)/(stp_j+sfp_j)
      var_rl <- r_l_bar*(1-r_l_bar)/(stp_l+sfn_l)
      var_ql <- q_l_bar*(1-q_l_bar)/(stp_l+sfp_l)
      rho_j <- cor(precision_j, recall_j)
      rho_l <- cor(precision_l, recall_l)
      
      var_j_wong[i] <- w_j^2*var_rj + (1-w_j)^2*var_qj + 2*w_j*(1-w_j)*rho_j*sqrt(var_rj)*sqrt(var_qj) 
      var_l_wong[i] <- w_l^2*var_rl + (1-w_l)^2*var_ql + 2*w_l*(1-w_l)*rho_l*sqrt(var_rl)*sqrt(var_ql) 
      var_jl_wong[i] <- var_j_wong[i] + var_l_wong[i]
      z_score_wong[i] <- (f_j[i] - f_l[i])/sqrt(var_jl_wong[i])
      
      if (is.na(var_jl_wong[i])||is.infinite(var_jl_wong[i])||is.na(var_jl[i])){
        idx <- c(idx, i)
        nremove = nremove + 1
      }
      if (is.na(var_jl_wong[i])||is.infinite(var_jl_wong[i])){
        nremove_w = nremove_w + 1
      }
      if (is.na(var_jl[i])){
        nremove_j = nremove_j + 1
      }
    }

    F_j <- F_j[-idx]
    F_l <- F_l[-idx]
    F_jl <- F_jl[-idx]
    f_j <- f_j[-idx]
    f_l <- f_l[-idx]
    var_j <- var_j[-idx]
    var_l <- var_l[-idx]
    var_jl <- var_jl[-idx]
    cov_F_jl <- cov_F_jl[-idx]
    corr_jl <- corr_jl[-idx]
    z_score_jl <- z_score_jl[-idx]
    var_j_wong <- var_j_wong[-idx]
    var_l_wong <- var_l_wong[-idx]
    var_jl_wong <- var_jl_wong[-idx]
    z_score_wong <- z_score_wong[-idx]
    
    F_j_mean <- mean(F_j)
    F_l_mean <- mean(F_l)
    F_jl_mean <- mean(F_jl)
    f_j_mean <- mean(f_j)
    f_l_mean <- mean(f_l)
    F_j_var <- var(F_j)
    F_l_var <- var(F_l)
    F_jl_var <- var(F_jl)
    corr_simu <- cor(F_j, F_l)
    z_score_simu <- F_jl_mean/sqrt(F_jl_var)
    
    var_j_mean <- mean(var_j)
    var_l_mean <- mean(var_l)
    cov_jl_mean <- mean(cov_F_jl)
    var_jl_mean <- mean(var_jl)
    corr_jl_mean <- mean(corr_jl)
    z_score_jvesr <- mean(z_score_jl)
    
    var_j_wong_mean <- mean(var_j_wong)
    var_l_wong_mean <- mean(var_l_wong)
    var_jl_wong_mean <- mean(var_jl_wong)
    z_score_wong <- mean(z_score_wong)
    
    return (list(nremove=nremove, nremove_j=nremove_j, nremove_w=nremove_w, f_j=F_j_mean, f_l=F_l_mean, var_j_simu=F_j_var, var_l_simu=F_l_var, corr_simu=corr_simu, var_jl_simu=F_jl_var, z_score_simu=z_score_simu, var_j_jvesr=var_j_mean, var_l_jvesr=var_l_mean, corr_jl_jvesr=corr_jl_mean, var_jl_jvesr=var_jl_mean, z_score_jvesr=z_score_jvesr, f_j_wong=f_j_mean, f_l_wong=f_l_mean, var_j_wong=var_j_wong_mean, var_l_wong=var_l_wong_mean, var_jl_wong=var_jl_wong_mean, z_score_wong=z_score_wong))
  }

main_abalone <-
  function(split, sd){
    # Before running the following codes, transfer the original downloaded data to columns
    Abalone <- read.table("data/abalone/abalone.txt")
    colnames(Abalone) <- c("Sex","Length", "Diam", "Height", "W_weight", "Sk_weight", "V_weight", "Sh_weight", "Rings")
    
    Abalone$Sex <- as.numeric(Abalone$Sex)
    # Abalone <- lapply(Abalone, function(x) as.numeric(as.character(x)))
    
    Abalone$Rings[Abalone$Rings < 6|Abalone$Rings > 6] <- 0
    Abalone$Rings[Abalone$Rings == 6] <- 1
    Abalone$Rings <- as.factor(as.character(Abalone$Rings))
    
    set.seed(sd)
    
    # Shuffle the original dataset by rows
    rows <- sample(nrow(Abalone))
    Abalone <- Abalone[rows, ] 
    
    train_data <- Abalone[1:split, ]
    test_data <- Abalone[(split+1):nrow(Abalone), ]
    
    return (list(train_data=train_data, test_data=test_data))
  }

a <- 0.5
b <- 0.5
group <- 5
seed <- 28
simu_time <- 1200 
test_size <- 1000 
split_abalone <- 2924

abalone_main = main_abalone(split_abalone, seed)
abalone_results = f_value_var(a, b, abalone_main$train_data, abalone_main$test_data, simu_time, test_size, group, seed)

