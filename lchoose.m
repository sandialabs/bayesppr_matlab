function out = lchoose(n, k)
    out = -1*betaln(1+n-k, 1+k) - log(n+1);
