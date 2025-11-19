function move_type = get_move_type(n_ridge, n_quant, n_ridge_max)

moves1 = ["death", "change"];
moves2 = ["birth", "death"];
moves3 = ["birth", "death", "change"];

if n_ridge == 0
    move_type = "birth";
elseif n_ridge==n_ridge_max
    if n_quant == 0
        move_type = "death";
    else
        tmp = randsample([1,2],1);
        move_type = moves1(tmp);
    end
else
    if n_quant == 0
        tmp = randsample([1,2],1);
        move_type = moves2(tmp);
    else
        tmp = randsample(1:3,1);
        move_type = moves3(tmp);
    end
end
