function [D, F, varargout] = bpdndf_dynamics_learning_behavior_x(data_obj, F_init, D_init, Psi_init, behavior, inf_opts)

% [D, F] = bpdndf_dynamics_learning(data_obj, F_init, D_init, inf_opts)
% 
% Function to learn both the representation and dynamics dictionaries for
% the BPDN-DF model.
% 
% 2018 - Adam Charles

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs

[data_type, dataShape, sig_opts] = checkInputTypes(data_obj);              % Check the input type and set up some parameters accordingly

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameter checking

inf_opts = check_inf_params(inf_opts);                                     % Check all the basic parameters
inf_opts = checkSizes(inf_opts, D_init, F_init, Psi_init, data_obj, data_type, sig_opts);       % Check the sizes of all the initializations % EY added data_obj, data_type, sig_opts 1/16/2026 in response to error
[D_true, F_true, Psi_true] = generateOptionalGT(data_type, sig_opts); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations

if isempty(F_init);   F = initialize_dynamics(inf_opts.nF, inf_opts.N);    % If necessary, initialize the dynamics
else;                 F = F_init;                                          % Otherwise just pass through
end
if isempty(D_init)
    if inf_opts.AcrossIndividuals % EY added 9/6/23
%         D = cell(size(data_obj,1),1);
        D = cellfun(@(x) initialize_dictionary(inf_opts.N, x),inf_opts.M,'UniformOutput',false);   % If necessary, initialize the dictionary
    else
        D = initialize_dictionary(inf_opts.N, inf_opts.M);   % If necessary, initialize the dictionary
    end
else;                 D = D_init;                                          % Otherwise just pass through
end

if isempty(Psi_init)
    if inf_opts.AcrossIndividuals % EY added 7/8/24
	    disp('FIXME: Psi_init AcrossIndividuals')
%         D = cell(size(data_obj,1),1);
        Psi = cellfun(@(x) initialize_dictionary(inf_opts.N, x),inf_opts.nBhv,'UniformOutput',false);   % If necessary, initialize the dictionary
    else
        Psi = initialize_dictionary(inf_opts.N, size(behavior{1},1));   % If necessary, initialize the dictionary
    end
else;                 Psi = Psi_init;                                          % Otherwise just pass through
end

% disp(size(D))
% disp('Initialize')
% disp(size(Psi))
% disp(size(behavior))
% pause()

clear D_init F_init Psi_init                                                       % Do some house cleaning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Learning

for n_iters = 1:inf_opts.max_iters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Make some data
    % disp(size(data_obj))
    % disp(size(behavior))
    

    if inf_opts.AcrossIndividuals % EY added 9/6/23, modified to use same indices for behavior 9/23/24
        [X_ex, idxStart1, idxEnd1] = cellfun(@(x) sampleSomeDataSeqs(inf_opts, data_type, sig_opts, x, ...
                                                           D_true, F_true),data_obj,'UniformOutput',false);
        [bhv_ex]                   = cellfun(@(x) sampleSomeDataSeqs(inf_opts, data_type, sig_opts, x, ...
                                                           D_true, F_true, idxStart1, idxEnd1),behavior,'UniformOutput',false); % EY added 7/9/24
    else
        [X_ex, idxStart1, idxEnd1] = sampleSomeDataSeqs(inf_opts, data_type, sig_opts, data_obj, ...
                                                           D_true, F_true);% Select some data for this learning iteration
        [bhv_ex]                   = sampleSomeDataSeqs(inf_opts, data_type, sig_opts, behavior, ...
                                                           D_true, F_true, idxStart1, idxEnd1); % EY added 7/9/24

    end
    disp("with matched sampling")
    % disp(idxEnd1)
    % disp(idxStart2)
    % disp(idxEnd2)

    % disp(X_ex(:,:))
    % disp(size(X_ex))
    % pause()
    % disp(bhv_ex(:,:))
    % disp(size(bhv_ex))
    % pause()
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Inference

    if inf_opts.AcrossIndividuals % EY added 08/27/2023
        for ii = 1:size(data_obj,1)
            [A_cell{ii},B_cell{ii}] = parallel_bilinear_dynamic_inference_behavior(X_ex{ii}, D{ii}, F, Psi{ii}, bhv_ex{ii}, ...
                                       @bpdndf_bilinear_handle_behavior_x, inf_opts); % Infer sparse coefficients
        end
    else
        [A_cell,B_cell] = parallel_bilinear_dynamic_inference_behavior(X_ex, D, F, Psi, bhv_ex, ...
                                       @bpdndf_bilinear_handle_behavior_x, inf_opts); % Infer sparse coefficients
    end
    % disp('Infer')
    % disp(size(A_cell))
    % disp(size(B_cell))
    % disp(A_cell)
    % disp(B_cell)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update step
    
    D_old = D;  F_old = F; Psi_old = Psi;                                  % Retain a copy of the representational & dynamics dictionary for comparison purposes
    % if (inf_opts.D_update == true) % (~strcmp(inf_opts.special, 'noobs'))||   % ASC added 7/25
    % 
    % 
    %     if inf_opts.AcrossIndividuals % EY added 08/27/2023
    %         for ii = 1:size(data_obj,1)                
    %             D{ii} = dictionary_update(cell2mat(X_ex{ii}), D{ii}, cell2mat(A_cell{ii}), ...
    %                                                      inf_opts.step_d); % Update static dictionary
    %         end
    %     else
    %         D = dictionary_update(cell2mat(X_ex), D, cell2mat(A_cell), ...
    %                                                      inf_opts.step_d); % Update static dictionary
    %     end
    % end  % ASC added 7/25

    if (inf_opts.D_update == true) % (~strcmp(inf_opts.special, 'noobs'))||   % ASC added 7/25
        if inf_opts.fewerneurpercol % EY added 01/16/2026
            dictionaryOpts.fewerneurpercol     = 1;
            dictionaryOpts.grad_type           = inf_opts.Dnorm; 
            disp(dictionaryOpts.grad_type)
            dictionaryOpts.nneg_dict           = 0;
            dictionaryOpts.lambda2             = inf_opts.lambda2;
            dictionaryOpts.GD_iters            = 1;
            dictionaryOpts.lambda_D            = inf_opts.lambda_D;
            dictionaryOpts.verysparsesuspected = 0;
            dictionaryOpts.fewerneurpercol     = 1;

            if inf_opts.AcrossIndividuals % EY added 08/27/2023
                for ii = 1:size(data_obj,1)
                    dictionaryOpts.in_iter = size(cell2mat(X_ex{ii}), 2);
                    D{ii} = dictionary_update(cell2mat(X_ex{ii}), D{ii}, cell2mat(A_cell{ii}), ...
                                                             inf_opts.step_d,...
                                                             dictionaryOpts); % Update static dictionary
                end
            else
                dictionaryOpts.in_iter     = size(cell2mat(X_ex), 2);
                D = dictionary_update(cell2mat(X_ex), D, cell2mat(A_cell), ...
                                                             inf_opts.step_d,...
                                                             dictionaryOpts); % Update static dictionary
            end
        else
            if inf_opts.AcrossIndividuals % EY added 08/27/2023
                for ii = 1:size(data_obj,1)                
                    D{ii} = dictionary_update(cell2mat(X_ex{ii}), D{ii}, cell2mat(A_cell{ii}), ...
                                                             inf_opts.step_d); % Update static dictionary
                end
            else
                D = dictionary_update(cell2mat(X_ex), D, cell2mat(A_cell), ...
                                                             inf_opts.step_d); % Update static dictionary
            end
        end
    end  % ASC added 7/25

    if inf_opts.fewerneurpercol
        inf_opts.lambda_D  = inf_opts.lambda_D * inf_opts.step_decay; % EY added 1/19/26
        disp('With lambda_D decay')
    end
    
    if (inf_opts.Psi_update == true) % (~strcmp(inf_opts.special, 'noobs'))||   % EY added 7/8/24
        if inf_opts.verysparsebhv
            dictionaryOpts.verysparsesuspected = 1;
            dictionaryOpts.in_iter             = size(cell2mat(bhv_ex), 2);
            dictionaryOpts.GD_iters            = 1;
            dictionaryOpts.grad_type           = inf_opts.psinorm; 
            disp(dictionaryOpts.grad_type)
            dictionaryOpts.nneg_dict           = 0;
            dictionaryOpts.lambda2             = inf_opts.lambda2;
            dictionaryOpts.fewerneurpercol     = 0; % not dealing with neurons in Psi

            if inf_opts.AcrossIndividuals 
                for ii = 1:size(data_obj,1)                
                    Psi{ii} = dictionary_update(cell2mat(bhv_ex{ii}), Psi{ii}, cell2mat(A_cell{ii}), ...
                                                             inf_opts.step_psi, ...
                                                             dictionaryOpts); % Update static dictionary
                end
            else
                Psi = dictionary_update(cell2mat(bhv_ex), Psi, cell2mat(A_cell), ...
                                                             inf_opts.step_psi,...
                                                             dictionaryOpts); % Update static dictionary
            end
        else
        
            if inf_opts.AcrossIndividuals 
                for ii = 1:size(data_obj,1)                
                    Psi{ii} = dictionary_update(cell2mat(bhv_ex{ii}), Psi{ii}, cell2mat(A_cell{ii}), ...
                                                             inf_opts.step_psi); % Update static dictionary
                end
            else
                Psi = dictionary_update(cell2mat(bhv_ex), Psi, cell2mat(A_cell), ...
                                                             inf_opts.step_psi); % Update static dictionary
            end
        end
    end  


    if inf_opts.AcrossIndividuals % EY added 08/27/2023
        A_cell_forF = reshape(vertcat(A_cell{:}),[],1).';
        B_cell_forF = reshape(vertcat(B_cell{:}),[],1).';
    end

    if (inf_opts.T_s > 1 && isempty(inf_opts.F_update)) || (inf_opts.F_update == true)
        if inf_opts.AcrossIndividuals
            F = dynamics_update(F, A_cell_forF, inf_opts.step_f, B_cell_forF, inf_opts.lambda_f);           % Update dynamics dictionary
        else
            F = dynamics_update(F, A_cell, inf_opts.step_f, B_cell, inf_opts.lambda_f);           % Update dynamics dictionary
        end
    end
    inf_opts.step_d   = inf_opts.step_d*inf_opts.step_decay;                 % Reduce the step size for the static dictionary
    inf_opts.step_f   = inf_opts.step_f*inf_opts.step_decay;                 % Reduce the step size for the dynamics dictionary
    inf_opts.step_psi = inf_opts.step_psi*inf_opts.step_decay;                 % Reduce the step size for the static dictionary

    if nargout > 2
        varargout{1} = Psi;
    end
    % disp('Update')
    % disp(size(Psi_old))
    % disp(size(Psi))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Display some statistics
    
    if inf_opts.AcrossIndividuals
        for ii = 1:size(data_obj,1)                
            tmp_plotting_fun(data_type,D{ii},F,inf_opts,n_iters, dataShape)
            f_err = 0;
            for mm = 1:numel(F)
                f_err = f_err + (1/numel(F))*sum(sum((F_old{mm} - F{mm}).^2))/sum(sum(F{mm}.^2));
            end
            fprintf('Iteration %d complete. Mean dD: %f, Mean dF: %f, Max b: %f, Median b use: %f, Mean dPsi: %f,\n,', n_iters, ...
                mean(sum((D{ii} - D_old{ii}).^2)./sum(D_old{ii}.^2)), f_err, max(max(abs(cell2mat(B_cell{ii})))), ...
                median(sum(abs(cell2mat(B_cell{ii}))>1e-3)),  mean(sum((Psi{ii} - Psi_old{ii}).^2)./sum(Psi_old{ii}.^2))   )
        end
    elseif strcmp(data_type, 'synth')
        tmp_plotting_fun(data_type,D,F,inf_opts,n_iters,D_true,F_true)
        f_err = correlate_learned_dynamics(F, F_true,D, dataShape);
        
        fprintf('Iteration %d complete. D error is %f and F error is %f (avg b is %f) and Psi error is %f\n', n_iters, ...
            sum((vec(D*(D')) - D_true(:)).^2), mean(f_err), mean(mean(cell2mat(B_cell))),sum((vec(Psi*(Psi')) - Psi_true(:)).^2) )
    elseif strcmp(data_type, 'BBC')||strcmp(data_type, 'datamatrix')
        tmp_plotting_fun(data_type,D,F,inf_opts,n_iters, dataShape)
        f_err = 0;
        for mm = 1:numel(F)
            f_err = f_err + (1/numel(F))*sum(sum((F_old{mm} - F{mm}).^2))/sum(sum(F{mm}.^2));
        end
        fprintf('Iteration %d complete. Mean dD: %f, Mean dF: %f, Max b: %f, Median b use: %f, Mean dPsi: %f\n', n_iters, ...
            mean(sum((D - D_old).^2)./sum(D_old.^2)), f_err, max(max(abs(cell2mat(B_cell)))), ...
            median(sum(abs(cell2mat(B_cell))>1e-3)) )
    elseif strcmp(data_type, 'datacell')             
        tmp_plotting_fun(data_type,D,F,inf_opts,n_iters, dataShape)
        f_err = 0;
        for mm = 1:numel(F)
            f_err = f_err + (1/numel(F))*sum(sum((F_old{mm} - F{mm}).^2))/sum(sum(F{mm}.^2));
        end
        fprintf('Iteration %d complete. Mean dD: %f, Mean dF: %f, Max b: %f, Median b use: %f, Mean dPsi: %f\n', n_iters, ...
            mean(sum((D - D_old).^2)./sum(D_old.^2)), f_err, max(max(abs(cell2mat(B_cell)))), ...
            median(sum(abs(cell2mat(B_cell))>1e-3)), mean(sum((Psi - Psi_old).^2)./sum(Psi_old.^2)) )
        
    else
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data_type, dataShape, sig_opts] = checkInputTypes(data_obj)

sig_opts = [];                                                             % Initialize option struct
if isstruct(data_obj)
    sig_opts  = data_obj;
    data_type = 'synth';
    dataShape = 'image';
elseif ischar(data_obj)
    data_type = data_obj;
    if strcmp(data_obj,'synth')
        sig_opts = default_synth_params();                                 % Get default parameters
    end
    dataShape = 'image';
elseif iscell(data_obj)
    data_type = 'datacell';
    if ismatrix(data_obj{1})
        dataShape = 'vector';
    elseif ndims(data_obj{1}) == 3
        dataShape = 'image';
    else
        error('Incompatiable number of dimensions in the data array!')
    end
elseif isnumeric(data_obj)
    data_type = 'datamatrix';
    if ismatrix(data_obj)
        dataShape = 'vector';
    elseif ndims(data_obj) == 3
        dataShape = 'image';
    else
        error('Incompatible number of dimensions in the data array!')
    end
    
elseif isempty(data_obj)
    data_type = 'synth';
    dataShape = 'image';
    sig_opts  = default_synth_params();
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function inf_opts = checkSizes(inf_opts, D_init, F_init, Psi_init, varargin) % EY added varargin 1/16/26

if nargin > 4
    data_obj  = varargin{1};
    data_type = varargin{2};
    sig_opts  = varargin{3};
end

if (~isfield(inf_opts,'M'))||isempty(inf_opts.M)
    if ~isempty(D_init)
        inf_opts.M = size(D_init,1);
    elseif strcmp(data_type,'datamatrix')
        inf_opts.M = size(data_obj,1);
    elseif isfield(sig_opts,'M')
        inf_opts.M = sig_opts.M;
    else
        inf_opts.M = 144;
    end
end
if (~isfield(inf_opts,'N'))||isempty(inf_opts.N)
    if ~isempty(D_init)
        inf_opts.N = size(D_init,2);
    elseif isfield(sig_opts,'N')
        inf_opts.N = sig_opts.N;
    else
        inf_opts.N = 4*inf_opts.M;
    end
end
if (~isfield(inf_opts,'nF'))||isempty(inf_opts.nF)
    if ~isempty(F_init)
        inf_opts.nF = numel(F_init);
    elseif isfield(sig_opts,'nF')
        inf_opts.nF = sig_opts.nF;
    else
        inf_opts.nF = 25;
    end
end

% check if this is useful
if (~isfield(inf_opts,'nBhv'))||isempty(inf_opts.nBhv)
    if ~isempty(Psi_init)
        inf_opts.nBhv = numel(Psi_init);
    elseif isfield(inf_opts,'nBhv')
        inf_opts.nBhv = inf_opts.nBhv;
    else
        inf_opts.nBhv = 1;
    end
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_ex, varargout] = sampleSomeDataSeqs(inf_opts, data_type, sig_opts, data_obj, D_true, F_true,varargin)
% modified to track which samples are used - EY 9/23/24

X_ex = cell(1,inf_opts.N_ex);

idxEnd   = [];
idxStart = [];

if nargin > 7
    idxEnd   = varargin{2};
    idxStart = varargin{1};
elseif nargin > 6
    idxStart = varargin{1};
end

if strcmp(data_type, 'synth')
    disp('not equipped to output indices for reuse')
    for kk = 1:inf_opts.N_ex
        [X_ex{kk}, ~, ~] = rand_seq_create(sig_opts, ...
                                         sig_opts.noise_opts, F_true); % Synthesize a new set of data examples
        X_ex{kk} = D_true*X_ex{kk}; 
    end
elseif strcmp(data_type, 'BBC')
    disp('not equipped to output indices for reuse')
    for kk = 1:inf_opts.N_ex
        X_ex{kk} = rand_bbc_video([sqrt(inf_opts.M), ... 
                     sqrt(inf_opts.M), inf_opts.T_s], inf_opts.dsamp); % Sample some videos from the BBC dataset
    end
elseif strcmp(data_type, 'datacell')||strcmp(data_type, 'datamatrix')
    [X_ex,idxStart,idxEnd] = sample_dynamic_exemplars(data_obj, inf_opts, idxStart, idxEnd);
else
    error('Unknown data type')
end

if nargout > 1
    varargout{1} = idxStart;
    varargout{2} = idxEnd;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D_true, F_true, Psi_true] = generateOptionalGT(data_type, sig_opts)
if strcmp(data_type, 'synth')
    F_true = rand_dyn_create(sig_opts.M, sig_opts.nF, 'perm');             % Create a dictionary of dynamics
    D_true = eye(sig_opts.M);                                              % Make a simple test case of a dictionary of points
    Psi_true = eye(sig_opts.nF);
else     
    F_true = []; D_true = []; Psi_true = [];
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%