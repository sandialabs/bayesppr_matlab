function alpha0 = get_log_mh_bd(n_ridge_prop, n_quant_prop, n_ridge_max)

if n_ridge_prop == 0
    alpha0 = 0;
elseif n_ridge_prop == n_ridge_max
    if n_quant_prop == 0
        alpha0 = 0;
    else 
        alpha0 = log(2);
    end
else
    if n_quant_prop == 0
        alpha0 = log(2);
    else
        alpha0 = log(3);
    end
end
