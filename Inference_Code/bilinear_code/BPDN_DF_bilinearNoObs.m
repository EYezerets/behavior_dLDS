function [varargout] = BPDN_DF_bilinearNoObs(varargin)
    
% [coef_dcs, recon_dcs, rMSE_dcs, PSNR_dcs] = ...
%        BPDN_DF_bilinear(MEAS_SIG, MEAS_SEL, DYN_FUN, DWTfunc, param_vals, ...
%        TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% DYN_FUN:    Tx1 or 1x1 cell array of the dynamics functions
% DWTfunc:    Wavelet transform (sparsity basis)
% param_vals: struct of parameter values (has fields: lambda_val (tradeoff
%             parameter for BPDN), lambda_history (tradeoff parameter
%             between prediction and data fidelity), and tol (tolerance for
%             TFOCS solver)) 
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% coef_dcs:  Nx1xT array of inferred sparse coefficients
% recon_dcs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% rMSE_dcs:  Tx1 array of rMSE values for the recovered video
% PSNR_dcs:  Tx1 array of PSNR values for the recovered video
% 
%
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 21, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs

MEAS_SIG   = varargin{1};
MEAS_FUN   = varargin{2};
DYN_FUN    = varargin{3};
DWTfunc    = varargin{4};
param_vals = varargin{5};

if nargin > 5
    rMSE_calc_opt = 1;
    TRUE_VID = varargin{6};
else
    rMSE_calc_opt = 0;
end

if isfield(param_vals, 'lambda_val')
    lambda_val = param_vals.lambda_val;
else
    lambda_val = 0.001;
end
if isfield(param_vals, 'lambda_history')
    lambda_history = param_vals.lambda_history;
else
    lambda_history = 0.2;
end
if isfield(param_vals, 'lambda_b')
    lambda_b = param_vals.lambda_b;
else
    lambda_b = 0.2;
end
if isfield(param_vals, 'lambda_historyb')
    lambda_historyb = param_vals.lambda_historyb;
else
    lambda_historyb = 0;
end
if isfield(param_vals, 'tol')
    TOL = param_vals.tol;
else
    TOL = 0.01;
end
if isfield(param_vals, 'verbose')
    verbose_flag = param_vals.verbose;
else
    verbose_flag = 0;
end
if isfield(param_vals, 'solver_type')
     doTFOCS = false;
     doFISTA = false;
     % doCVX   = false;

     if strcmp(param_vals.solver_type, 'tfocs')
         doTFOCS = true;
     end
     if strcmp(param_vals.solver_type, 'fista')
         doFISTA = true;
     end
     % if strcmp(param_vals.solver_type, 'cvx')
     %     doCVX   = true;
     % end
else
    doTFOCS = true;
    doFISTA = false;
    % doCVX   = false;
end
if isfield(param_vals, 'deltaDynamics');    deltaOpt = param_vals.deltaDynamics;
else;                                       deltaOpt = false;
end

DWT_apply  = DWTfunc.apply;
DWT_invert = DWTfunc.invert;

meas_func = MEAS_FUN{1};
Phit      = meas_func.Phit; 

M   = numel(MEAS_SIG(:, :, 1));                         % size of measured signal
N2  = numel(DWT_apply(Phit(MEAS_SIG(:, :, 1))));        % size of uncompressed signal (a)
N_b = numel(DYN_FUN);                                   % number of dynamics functions

opts.tol        = TOL;
opts.printEvery = 0;
num_frames      = size(MEAS_SIG, 3);
if isreal(MEAS_SIG)
    opt_set = 'R2R';
elseif ~isreal(MEAS_SIG)
    opt_set = 'C2C';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for initial frame

meas_func = MEAS_FUN{1};
Phi       = meas_func.Phi;
Phit      = meas_func.Phit; 

Af = @(x) Phi(DWT_invert(x));
Ab = @(x) DWT_apply(Phit(x));
A  = linop_handles([M, N2], Af, Ab, opt_set);

% res    = solver_L1RLS(A, MEAS_SIG(:, :, 1), lambda_val, zeros(N2, 1), opts );
% im_res = DWT_invert(real(res));

% Save reconstruction results
% coef_dcs(:, :, 1)  = res;
coef_dcs(:,:,1)    = MEAS_SIG(:, :, 1);
bcoef_dcs(:,:,1)   = zeros(N_b,1);
recon_dcs = [];
% recon_dcs(:, :, 1) = im_res;
tic
% if rMSE_calc_opt == 1
%     rMSE_dcs(1) = sum(sum((recon_dcs(:, :, 1) - TRUE_VID(:, :, 1)).^2))/sum(sum(TRUE_VID(:, :, 1).^2));
%     PSNR_dcs(1) = psnr(real(recon_dcs(:, :, 1)), TRUE_VID(:, :, 1), 2);
%     TIME_ITER   = toc;
%     if verbose_flag == 1
%         fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', 1, num_frames, TIME_ITER, PSNR_dcs(1), rMSE_dcs(1))
%     end
% else
%     TIME_ITER = toc;
%     if verbose_flag == 1
%         fprintf('Finished frame %d of %d in %f seconds. \n', 1, num_frames, TIME_ITER)
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for rest of frames
optsreset = opts;

for kk = 2:num_frames % Can probably make this a parfor now...
    tic

    if deltaOpt;  yNow = MEAS_SIG(:,:,kk) - MEAS_SIG(:,:,kk-1);
    else;         yNow = MEAS_SIG(:,:,kk);                        end

    
    % Get the Dynamics function
    f_dyn = zeros(size(coef_dcs, 1),numel(DYN_FUN));
    for ll = 1:N_b
        f_dyn(:, ll) = DYN_FUN{ll}*MEAS_SIG(:, :, kk-1);                    % Create what is essentially the dictionary with respect to the dynamics coefficients  
    end
    
    % Get the Measurement function
    if numel(MEAS_FUN) == 1
        meas_func = MEAS_FUN{1};
    elseif numel(MEAS_FUN) == num_frames
        meas_func = MEAS_FUN{kk};
    else
        error('You need either the same measurement function for all time or one dynamics function per time-step!')
    end

    % Set  up A and At for TFOCS
    Phi  = meas_func.Phi;
    Phit = meas_func.Phit;

    if doFISTA
        if lambda_historyb == 0
            
            opts.pos        = false;
            opts.lambda     = lambda_b*ones(N_b,1);
            opts.check_grad = 0;
    
            res = fista_lasso(yNow, f_dyn, [], opts);
    
        else
            % disp(f_dyn)
            A      = [sqrt(lambda_historyb)*eye(N_b,N_b); f_dyn];
            Y      = [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow];
            lamVec = lambda_b*ones(N_b,1);
    
            opts.pos        = false;
            opts.lambda     = lamVec(:);
            opts.check_grad = 0;
            res = fista_lasso(Y, A, [], opts);

            opts.lambda = lamVec(:)./(1 + 200*abs(res));
            res         = fista_lasso(Y, A, [], opts);

        end
    end
    
    if doTFOCS
        if lambda_historyb == 0
            
    %         Af = @(x) f_dyn*x;
    %         Ab = @(x) (f_dyn')*x;
    %         
    %         A = linop_handles([M,M],Af,Ab,opt_set);
            if doFISTA == 1
                opts   = optsreset; % when you run both 
            end
            % only needed one reg parameter (lambda_b in this case) bc only two terms in
            % optimization
            res = solver_L1RLS(f_dyn,yNow,lambda_b,zeros(N_b,1),opts); % replaced MEAS_SIG(:, :, kk) with yNow
        else
            % warning('no current implementation of lambda_historyb > 0; res will be set to zero')
            % res = zeros(M,1);

            Af = @(x) [sqrt(lambda_historyb)*x(1:N_b); 
                f_dyn*x(1:N_b)]; 
            Ab = @(x) (sqrt(lambda_historyb)*x(1:N_b) + ...
                (f_dyn')*x((N_b+1):(N_b+N2))); % sign matches elements of Y
            if doFISTA == 1
                opts   = optsreset; % when you run both 
            end
    
            % A = linop_handles([N2+N_b+size(thisBehaviorSample,1), N2+N_b], Af, Ab, opt_set); % A transpose
            A = linop_handles([N_b+N2, N_b], Af, Ab, opt_set); % A transpose
            % Optimize the BPDN objective function with TFOCS
            lamVec = lambda_b*ones(N_b,1);
            % lamVec = [lambda_val*ones(M,1);lambda_b*ones(N_b,1)];
            res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow], lamVec, zeros(N_b, 1), opts);

            
        %     Af = @(x) [sqrt(lambda_historyb)*x(1:N_b); ...
        %            sqrt(lambda_history)*x(N_b+1:end) - sqrt(lambda_history)*f_dyn*x(1:N_b)]; % N_b
        %     Ab = @(x) (sqrt(lambda_history)*x((N_b+1):end) + ...
        %         sqrt(lambda_historyb)*x(1:N_b) - sqrt(lambda_history)*(f_dyn')*x((N_b+1):end)); 
        %     A = linop_handles([M+N_b, N_b], Af, Ab, opt_set);
        %     % Optimize the BPDN objective function with TFOCS
        %     % lamVec = [lambda_val*ones(M,1);lambda_b*ones(N_b,1)];
        %     lamVec = lambda_b*ones(N_b,1);
        % %     fprintf('condition of F is %f\n', cond(f_dyn))
        %     if doFISTA == 1
        %         opts   = optsreset; % when you run both 
        %     end
        %     res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow], lamVec, zeros(N_b, 1), opts );

            
        %     Af = @(x) (sqrt(lambda_historyb)*x(1:N_b) + sqrt(lambda_history)*yNow - sqrt(lambda_history)*f_dyn*yNow); % N_b
        %     Ab = @(x) (sqrt(lambda_history)*yNow + sqrt(lambda_historyb)*x(1:N_b) - sqrt(lambda_history)*(f_dyn')*yNow); 
        %     A = linop_handles([N_b, N_b], Af, Ab, opt_set);
        %     % Optimize the BPDN objective function with TFOCS
        %     lamVec = lambda_b*ones(N_b,1);
        % %     fprintf('condition of F is %f\n', cond(f_dyn))
        %     if doFISTA == 1
        %         opts   = optsreset; % when you run both 
        %     end
        %     res    = solver_L1RLS(A, sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1), lamVec, zeros(N_b, 1), opts );


        %     Af = @(x) [sqrt(lambda_historyb)*x(1:N_b); ...
        %            sqrt(lambda_history)*x(N_b+1:end) - sqrt(lambda_history)*f_dyn*x(1:N_b)]; % N_b
        %     Ab = @(x) (sqrt(lambda_history)*x((N_b+1):end); ...
        %            sqrt(lambda_historyb)*x(1:N_b) - sqrt(lambda_history)*(f_dyn')*x((N_b+1):end)); 
        %     A = linop_handles([M+N_b, M+N_b], Af, Ab, opt_set);
        %     % Optimize the BPDN objective function with TFOCS
        %     lamVec = [lambda_val*ones(M,1);lambda_b*ones(N_b,1)];
        % %     fprintf('condition of F is %f\n', cond(f_dyn))
        %     if doFISTA == 1
        %         opts   = optsreset; % when you run both 
        %     end
        %     res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow], lamVec, zeros(M+N_b, 1), opts );



            % Af = @(x) (f_dyn*x; );
            % Ab = @(x) (f_dyn')*x;
            % 
            % A = linop_handles([M,M],Af,Ab,opt_set);
            % 
            % if doFISTA == 1
            %     opts   = optsreset; % when you run both 
            % end
            % res = solver_L1RLS(f_dyn, [MEAS_SIG(:,:,kk); bcoefs(:,:,kk),lambda_b,zeros(N_b,1),opts);
            % 
        end
    end

    coef_dcs(:,:,kk)  = MEAS_SIG(:,:,kk);
    bcoef_dcs(:,:,kk) = res; % for tfocs, but also works for fista? res(1:N_b,:);

%     if lambda_historyb == 0
%         if lambda_val <= lambda_b
%             gamma_hat = lambda_history*(lambda_val/lambda_b);
%             Af  = @(x) [lambda_history*DWT_invert(x(1:N2)) - gamma_hat*f_dyn*x(N2+1:N2+N_b); ...
%                                                            DWT_invert(x(1:N2))];
%             Ab  = @(x) [DWT_apply(x(N2+1:end)) + lambda_history*DWT_apply(x(1:N2));...
%                                                         -gamma_hat*(f_dyn')*x(1:N2)];
%             A   = linop_handles([M+M, M+N_b], Af, Ab, opt_set);
%             % Optimize the BPDN objective function with TFOCS
%             res = solver_L1RLS( A, [zeros(N2,1); MEAS_SIG(:, :, kk)], lambda_val, zeros(M+N_b, 1), opts );
%         elseif lambda_val > lambda_b
%             lam_rat   = (lambda_b/lambda_val);
%             gamma_hat = lambda_history*(lambda_b/lambda_val);
%             Af = @(x) [gamma_hat*DWT_invert(x(1:N2)) - lambda_history*f_dyn*x(N2+1:N2+N_b); ...
%                                                     lam_rat*DWT_invert(x(1:N2))];
%             Ab = @(x) [lam_rat*DWT_apply(x(N2+1:end)) + gamma_hat*DWT_apply(x(1:N2));...
%                                                     -lambda_history*(f_dyn')*x(1:N2)];
%             A  = linop_handles([M+M, N2+N_b], Af, Ab, opt_set);
%             % Optimize the BPDN objective function with TFOCS
%             res = solver_L1RLS( A, [zeros(N2,1); MEAS_SIG(:, :, kk)], lambda_b, zeros(M+N_b, 1), opts );
%         end
%     else
%         if lambda_val <= lambda_b
%             gamma_hat = lambda_history*(lambda_val/lambda_b);
%             lb_h      = lambda_historyb*(lambda_val/lambda_b);
%             Af = @(x) [lb_h*x(M+1:M+N_b); ...
%                    lambda_history*DWT_invert(x(1:N2)) - gamma_hat*f_dyn*x(N2+1:M+N_b); ...
%                    DWT_invert(x(1:M))];
%             Ab = @(x) [DWT_apply(x((N_b+M+1):end)) + lambda_history*DWT_apply(x((N_b+1):(N_b+M))); ...
%                    lb_h*x(1:N_b) - gamma_hat*(f_dyn')*x((N_b+1):(N_b+M))];
%             A  = linop_handles([M+M+N_b, N2+N_b], Af, Ab, opt_set);
%             % Optimize the BPDN objective function with TFOCS
%             res = solver_L1RLS( A, [lambda_historyb*bcoef_dcs(:,:,kk-1); zeros(N2,1); MEAS_SIG(:, :, kk)], lambda_val, zeros(M+N_b, 1), opts );
%         elseif lambda_val > lambda_b
%             lam_rat = (lambda_b/lambda_val);
%             gamma_hat = lambda_history*(lambda_b/lambda_val);
%             Af = @(x) [lambda_historyb*x(M+1:M+N_b); ...
%                    gamma_hat*DWT_invert(x(1:M)) - lambda_history*f_dyn*x(M+1:M+N_b); ...
%                    lam_rat*Phi(DWT_invert(x(1:M)))];
%             Ab = @(x) [lam_rat*DWT_apply(Phit(x((N_b+M+1):end))) + gamma_hat*DWT_apply(x((N_b+1):(N_b+M))); ...
%                    lambda_historyb*x(1:N_b) - lambda_history*(f_dyn')*x((N_b+1):(N_b+M))];
%             A = linop_handles([M+M+N_b, N2+N_b], Af, Ab, opt_set);
%             % Optimize the BPDN objective function with TFOCS
%             res = solver_L1RLS(A, [lambda_historyb*bcoef_dcs(:,:,kk-1); zeros(N2,1); MEAS_SIG(:, :, kk)], lambda_b, zeros(M+N_b, 1), opts );
%         end
%     end

%     im_res = DWT_invert(real(res(1:M)));

    % Save reconstruction results
%     if lambda_val <= lambda_b
%         coef_dcs(:, :, kk) = res(1:M);
%         bcoef_dcs(:, :, kk) = (lambda_val/lambda_b)*res(M+1:M+N_b);
%     elseif lambda_val > lambda_b
%         coef_dcs(:, :, kk) = (lambda_b/lambda_val)*res(1:M);
%         bcoef_dcs(:, :, kk) = res(M+1:M+N_b);
%     end
%    
%    bcoef_dcs(:,:,kk) = res;
%    
%     recon_dcs(:, :, kk) = im_res;
%     if rMSE_calc_opt == 1
%         rMSE_dcs(kk) = sum(sum((recon_dcs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
%         PSNR_dcs(kk) = psnr(real(recon_dcs(:, :, kk)), TRUE_VID(:, :, kk), 2);
%         TIME_ITER = toc;
%         if verbose_flag == 1
%             fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, PSNR_dcs(kk), rMSE_dcs(kk))
%         end
%     else
%         TIME_ITER = toc;
%         if verbose_flag == 1
%             fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
%         end
%     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

if (rMSE_calc_opt == 1)
    if nargout > 0
        varargout{1} = {coef_dcs, bcoef_dcs};
    end
    if nargout > 1
        varargout{2} = recon_dcs;
    end
    if nargout > 2
        varargout{3} = rMSE_dcs;
    end
    if nargout > 3
        varargout{4} = PSNR_dcs;
    end
    if nargout > 4
        for kk = 5:nargout
            varargout{kk} = [];
        end
    end
elseif (rMSE_calc_opt ~= 1)
    if nargout > 0
        varargout{1} = {coef_dcs, bcoef_dcs};
    end
    if nargout > 1
        varargout{2} = recon_dcs;
    end
    if nargout > 2
        for kk = 3:nargout
            varargout{kk} = [];
        end
    end
else
    error('How did you get here?')
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
