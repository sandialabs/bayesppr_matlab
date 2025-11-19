classdef PoolBPPRPredict
    % class for parallel BASS

    properties
        X
        mcmc_use
        bm_list
    end

    methods
        function obj = PoolBPPRPredict(X, mcmc_use, bm_list)
            obj.X = X;
            obj.mcmc_use = mcmc_use;
            obj.bm_list = bm_list;
        end

        function pred = listpredict(obj, i)
            pred = obj.bm_list(i).predict(obj.X, obj.mcmc_use);
        end

        function out = predict(obj, ncores, nlist)
            if isempty(gcp('nocreate'))
                parpool(ncores);
            end
            out = cell(1,nlist);
            bar = ProgressBar(nrow_y, ...
                'IsParallel', true, ...
                'WorkerDirectory', pwd, ...
                'Title', 'Running MCMC Chains' ...
                );
            bar.setup([], [], []);
            parfor i = 1:nlist
                out{i} = obj.listpredict(i);
                updateParallel([], pwd);
            end
            bar.release();
        end
    end
end
