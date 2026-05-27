function [varargout] = BPDN_DF_bilinearNoObs_behavior(varargin)
    
% [coef_dcs, recon_dcs, rMSE_dcs, PSNR_dcs] = ...
%        BPDN_DF_bilinear(MEAS_SIG, MEAS_FUN, DYN_FUN, DWTfunc, param_vals, ...
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
Psi        = varargin{6};
behavior   = varargin{7};

if nargin > 5
    rMSE_calc_opt = 1;
else
    rMSE_calc_opt = 0;
end

if isfield(param_vals, 'lambda_b')
    lambda_b = param_vals.lambda_b;
else
    lambda_b = 0.2;
end
if isfield(param_vals, 'lambda_history')
    lambda_history = param_vals.lambda_history;
else
    lambda_history = 0;
end
if isfield(param_vals, 'lambda_historyb')
    lambda_historyb = param_vals.lambda_historyb;
else
    lambda_historyb = 0;
end
if isfield(param_vals, 'tol');    TOL = param_vals.tol;
else;                             TOL = 0.01;
end
if isfield(param_vals, 'deltaDynamics');    deltaOpt = param_vals.deltaDynamics;
else;                                       deltaOpt = false;
end

if isfield(param_vals, 'lambda_behavior')
    lambda_behavior = param_vals.lambda_behavior;
else
    lambda_behavior = 0.1;
end

if isfield(param_vals, 'nBhv')
    nBhv = param_vals.nBhv;
else
    nBhv  = size(Psi,1);
end
if isreal(MEAS_SIG)
    opt_set = 'R2R';
elseif ~isreal(MEAS_SIG)
    opt_set = 'C2C';
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

% if isfield(param_vals, 'CVX_Precision')
%     CVX_Precision = param_vals.CVX_Precision;
% else
%     CVX_Precision = 'default';
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for initial frame

% Save reconstruction results
coef_dcs(:, :, 1)  = MEAS_SIG(:, :, 1);
bcoef_dcs(:,:,1)   = zeros(N_b,1);
recon_dcs          = [];



PsiFunc       = meas_func.Psi;
PsitFunc      = meas_func.Psit; 

APsif = @(x) PsiFunc(DWT_invert(x));
% APsib = @(x) DWT_apply(Psit(x));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for rest of frames
whatIsPsi = APsif(eye(N_b));

% thisBehaviorSample = behavior{:,:};
thisBehaviorSample = squeeze(behavior);
if size(thisBehaviorSample,2) == 1
    thisBehaviorSample = thisBehaviorSample.'; %rows are now dims, cols are time points (to match multidimensional behavior after squeeze)
end

optsreset = opts;

for kk = 2:num_frames % Can probably make this a parfor now...
    tic
    
    % Get the Dynamics function
    f_dyn = zeros(size(coef_dcs, 1),numel(DYN_FUN));
    for ll = 1:N_b
        f_dyn(:, ll) = DYN_FUN{ll}*MEAS_SIG(:, :, kk-1);                    % Create what is essentially the dictionary with respect to the dynamics coefficients  
    end
    
    % Get the Measurement function
    if numel(MEAS_FUN) == 1;               meas_func = MEAS_FUN{1};
    elseif numel(MEAS_FUN) == num_frames;  meas_func = MEAS_FUN{kk};
    else; error('You need either the same measurement function for all time or one dynamics function per time-step!')
    end

    
    
    if deltaOpt;  yNow = MEAS_SIG(:,:,kk) - MEAS_SIG(:,:,kk-1);
    else;         yNow = MEAS_SIG(:,:,kk);                        end

    if doTFOCS

        Af = @(x) [sqrt(lambda_historyb)*x(1:N_b); 
            f_dyn*x(1:N_b);
            sqrt(lambda_behavior)*PsiFunc(DWT_invert(x(1:N_b)))]; 
        Ab = @(x) (sqrt(lambda_historyb)*x(1:N_b) + ...
            (f_dyn')*x((N_b+1):(N_b+N2)) + ...
            sqrt(lambda_behavior)*DWT_apply(PsitFunc(x(N_b+N2+1:end)))); % sign matches elements of Y
        if doFISTA == 1
            opts   = optsreset; % when you run both 
        end

        A = linop_handles([N_b+N2+size(thisBehaviorSample,1), N_b], Af, Ab, opt_set); % A transpose
        % Optimize the BPDN objective function with TFOCS
        lamVec = lambda_b*ones(N_b,1);
        res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow; sqrt(lambda_behavior)*thisBehaviorSample(:,kk)], lamVec, zeros(N_b, 1), opts);

        % Af = @(x) [sqrt(lambda_historyb)*x(N2+1:N2+N_b); ...
        %             - sqrt(lambda_history)*f_dyn*x(N2+1:N2+N_b); ...
        %             PsiFunc(DWT_invert(x(N2+1:N2+N_b)))];
        % Ab = @(x) (sqrt(lambda_history)*x((N_b+1):(N_b+N2)) + ...
        %     DWT_apply(PsitFunc(x((N_b+N2+1):end))) + ...
        %     sqrt(lambda_historyb)*x(1:N_b) - ...
        %     sqrt(lambda_history)*(f_dyn')*x((N_b+1):(N_b+N2)));
        % if doFISTA == 1
        %     opts   = optsreset; % when you run both 
        % end
        % 
        % % A = linop_handles([N2+N_b+size(thisBehaviorSample,1), N2+N_b], Af, Ab, opt_set); % A transpose
        % A = linop_handles([N_b+N2+size(thisBehaviorSample,1), N_b], Af, Ab, opt_set); % A transpose
        % % Optimize the BPDN objective function with TFOCS
        % lamVec = lambda_b*ones(N_b,1);
        % res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow; sqrt(lambda_behavior)*thisBehaviorSample(:,kk-1)], lamVec, zeros(N_b, 1), opts);

    %     Af = @(x) [sqrt(lambda_historyb)*x(N2+1:N2+N_b); ...
    %                 - sqrt(lambda_history)*f_dyn*x(N2+1:N2+N_b); ...
    %                 eye(N2,N2);
    %                 PsiFunc(DWT_invert(x(N2+1:N2+N_b)))];
    %     Ab = @(x) [sqrt(lambda_history)*x((N_b+1):(N_b+N2)) + eye(N2,N2); ...
    %                DWT_apply(PsitFunc(x((N_b+N2+size(yNow,1)+1):end))) + sqrt(lambda_historyb)*x(1:N_b) - sqrt(lambda_history)*(f_dyn')*x((N_b+1):(N_b+N2))];
    %     if doFISTA == 1
    %         opts   = optsreset; % when you run both 
    %     end
    % 
    %     A = linop_handles([M+N2+N_b+size(thisBehaviorSample,1), N2+N_b], Af, Ab, opt_set); % A transpose
    %     % Optimize the BPDN objective function with TFOCS
    %     lamVec = lambda_b*ones(N_b,1);
    % %     fprintf('condition of F is %f\n', cond(f_dyn))
    % 
    %     res    = solver_L1RLS(A, [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow; zeros(N2,1); sqrt(lambda_behavior)*thisBehaviorSample(:,kk-1)], lamVec, zeros(N2+N_b, 1), opts);
    %     % resTFOCS(:,kk) = res;
    end

    if doFISTA

        A      = [sqrt(lambda_historyb)*eye(N_b,N_b); f_dyn; sqrt(lambda_behavior)*whatIsPsi];
        Y      = [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow; sqrt(lambda_behavior)*thisBehaviorSample(:,kk)];
        lamVec = lambda_b*ones(N_b,1);

        opts.pos        = false;
        opts.lambda     = lamVec(:);
        opts.check_grad = 0;
        res = fista_lasso(Y, A, [], opts);
        opts.lambda = lamVec(:)./(1 + 200*abs(res));
        res         = fista_lasso(Y, A, [], opts);

    %     if lambda_historyb == 0
    % 
    %         A      = [sqrt(lambda_historyb)*eye(N_b,N_b); f_dyn; sqrt(lambda_behavior)*whatIsPsi];
    %         Y      = [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow];
    %         lamVec = lambda_b*ones(N_b,1);
    % 
    %         opts.pos        = false;
    %         opts.lambda     = lamVec(:);
    %         opts.check_grad = 0;
    %         res = fista_lasso(Y, A, [], opts);
    % 
    % %         A = linop_handles([M,M],Af,Ab,opt_set);
    % %         res = solver_L1RLS(f_dyn, yNow,lambda_b,zeros(N_b,1),opts);
    %     else
    % 
    %         A      = [sqrt(lambda_historyb)*eye(N_b,N_b); f_dyn; sqrt(lambda_behavior)*whatIsPsi];
    %         Y      = [sqrt(lambda_historyb)*bcoef_dcs(:,:,kk-1); yNow; thisBehaviorSample(:,kk-1)];
    %         lamVec = lambda_b*ones(N_b,1);
    % 
    %         opts.pos        = false;
    %         opts.lambda     = lamVec(:);
    %         opts.check_grad = 0;
    %         res = fista_lasso(Y, A, [], opts);
    % %         res = res + sign(res).*opts.lambda.*(abs(res) > 0.1*max(abs(res)));% Correct lasso bias
    %         % OPTION 1
    % %         idxNNZ = (abs(res) > 0.5*max(abs(res)));
    % %         res(~idxNNZ) = 0;
    % %         res(idxNNZ)  = A(:,idxNNZ)\Y;
    %         % OPTION 2
    % %         opts.lambda = lamVec(:).*(~idxNNZ);
    % %         opts.lambda = lamVec(:)./(1 + 200*abs(res));
    %         opts.lambda = lamVec(:)./(1 + 200*abs(res));
    %         res         = fista_lasso(Y, A, [], opts);
    %     end
    end

    coef_dcs(:,:,kk)  = MEAS_SIG(:,:,kk);
    bcoef_dcs(:,:,kk) = res;
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
