classdef PoolBPPR
    % class for parallel BASS

    properties
        x
        y
        opts
    end

    methods
        function obj = PoolBPPR(x, y, opts)
            obj.x = x;
            obj.y = y;
            obj.opts = opts;
            obj.opts.silent = true;   % the pool reports progress via ProgressBar
        end

        function bm = rowbppr(obj, i)
            % bppr takes name-value options, so expand the struct rather than
            % passing the fields positionally.
            bppr_args = namedargs2cell(obj.opts);
            bm = bppr(obj.x, obj.y(i,:)', bppr_args{:});
        end

        function out = fit(obj, ncores, nrow_y)
            if isempty(gcp('nocreate'))
                parpool(ncores);
            end
            out = cell(1,nrow_y);
            bar = ProgressBar(nrow_y, ...
                'IsParallel', true, ...
                'WorkerDirectory', pwd, ...
                'Title', 'Running MCMC Chains' ...
                );
            bar.setup([], [], []);
            parfor i = 1:nrow_y
                out{i} = obj.rowbppr(i);
                updateParallel([], pwd);
            end
            bar.release();
        end
    end
end
