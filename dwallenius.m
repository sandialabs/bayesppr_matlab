function ss = dwallenius(wfeat_norm, feat)
% Multivariate Walenius' noncentral hypergeometric density function with some variables fixed

if (length(feat) == length(wfeat_norm))
    ss = 1;
else
    logMH = wfeat_norm(feat) ./ (1 - sum(wfeat_norm(feat)));

    j = length(logMH);
    ss = 1 + (-1)^j / (sum(logMH) + 1);
    for i = 1:(j-1)
        idx = nchoosek(1:j,i+1);
        temp = logMH(idx);
        ss = ss + (-1)^(i) * sum(1./(sum(temp,2)+ 1));
    end
end
