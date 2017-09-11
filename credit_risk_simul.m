    %% Winter 2017 CFRM, Taoxin JIANG, 998874231 Assignment 3
    clear; close all;clc;format long;

    Nout  = 100000; % number of out-of-sample scenarios
    Nin   = 5000;   % number of in-sample scenarios
    Ns    = 5;      % number of idiosyncratic scenarios for each systemic

    C = 8;          % number of credit states

    % Filename to save out-of-sample scenarios
    filename_save_out  = 'scen_out';

    % Read and parse instrument data
    instr_data = dlmread('instrum_data.csv', ',');
    instr_id   = instr_data(:,1);           % ID
    driver     = instr_data(:,2);           % credit driver
    beta       = instr_data(:,3);           % beta (sensitivity to credit driver)
    recov_rate = instr_data(:,4);           % expected recovery rate
    value      = instr_data(:,5);           % value
    prob       = instr_data(:,6:6+C-1);     % credit-state migration probabilities (default to A)
    % for all counterparties, the probabilities to change credit-state
    exposure   = instr_data(:,6+C:6+2*C-1); % credit-state migration exposures (default to A)
    % for all counterparties, the exposures to change credit-state
    retn       = instr_data(:,6+2*C);       % market returns

    %disp (exposure)
    K = size(instr_data, 1); % number of  counterparties

    % Read matrix of correlations for credit drivers
    rho = dlmread('credit_driver_corr.csv', '\t');
    sqrt_rho = (chol(rho))'; % Cholesky decomp of rho (for generating correlated Normal random numbers)

    disp('======= Credit Risk Model with Credit-State Migrations =======')
    disp('============== Monte Carlo Scenario Generation ===============')
    disp(' ')
    disp(' ')
    disp([' Number of out-of-sample Monte Carlo scenarios = ' int2str(Nout)])
    disp([' Number of in-sample Monte Carlo scenarios = ' int2str(Nin)])
    disp([' Number of counterparties = ' int2str(K)])
    disp(' ')

    % Find credit-state for each counterparty
    % 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
    [Ltemp, CS] = max(prob, [], 2); % index of max prob to migrate credit state
    clear Ltemp

    % Account for default recoveries
    exposure(:, 1) = (1-recov_rate) .* exposure(:, 1);

    % Compute credit-state boundaries
    CS_Bdry = norminv( cumsum(prob(:,1:C-1), 2) );
    % cumsum find the cirmulative probability, norminv find the interval x that
    % lies in the cirmulated probability, for each counterparties (100) for
    % each cirmulated sections (7)

    % -------- Insert your code here -------- %
    loss_s = zeros(100,Nout); % create a variable to store loss
    if(~exist('scenarios_out.mat','file')) 
        % -------- Insert your code here -------- %
        for s = 1:Nout
            y_sim = sqrt_rho * (normrnd(0,1,1,size(rho,1)))'; % systemic loss (co-related random variable)
            by_sim = beta.* y_sim(driver); % 50*1 % expanded from driver to counterparty and multiply by beta 
            % -------- Insert your code here -------- %
            sim_prob = by_sim + sqrt(1-beta.^2).*normrnd(0,1,K,1); % create independent random variable
            % Identify the interval (please see report for details)
            sim_prob_dup = repmat(sim_prob,1,size(CS_Bdry,2)); % create duplicates of random variables
            sim_diff = -sim_prob_dup + CS_Bdry; % find the difference between variables and CS_bdry
            sim_bool = (sim_diff) >= 0; 
            sim_bool_sum = sum(sim_bool,2); 
            sim_interval_1 = C - sim_bool_sum; % found the interval
            loss_s_temp = zeros(K,1);
            for i = 1:size(CS_Bdry,1)
                loss_s_temp(i) = exposure(i,sim_interval_1(i)); % using the interval to find the losses of each asset
            end
            loss_s(:,s) = loss_s_temp;
        end

        % Calculated out-of-sample losses (100000 x 100)
        % Losses_out
        Losses_out = loss_s'; % Save
        save('scenarios_out', 'Losses_out')
    else
        load('scenarios_out', 'Losses_out')
    end

    % Normal approximation computed from out-of-sample scenarios
    mu_l = mean(Losses_out)'; % find the mean of asset loss
    var_l = cov(Losses_out); % find the covariance matrix of asset loss
    % Verfication of Monte Carlo for Out-of-Sample
    ver = diag(prob*exposure'); % Using probability expectation to verify the losses
    diff_ver_percent = mean(abs(ver-mu_l)./ver*100); % percent of difference

    % Compute portfolio weights
    portf_v = sum(value);     % portfolio value
    w0{1} = value / portf_v;  % asset weights (portfolio 1)
    w0{2} = ones(K, 1) / K;   % asset weights (portfolio 2)
    x0{1} = (portf_v ./ value) .* w0{1};  % asset units (portfolio 1)
    x0{2} = (portf_v ./ value) .* w0{2};  % asset units (portfolio 2)
    pf1 = x0{1}'*value; %value of pf1
    pf2 = x0{2}'*value; %value of pf2
    disp (['The portfoilo 1 value is $', num2str(pf1)])
    disp (['The portfoilo 2 value is $', num2str(pf2)])
    % Quantile levels (99%, 99.9%)
    alphas = [0.99 0.999];

    % Initialize the result variables
    VaRout = zeros (2,2);
    VaRinN = zeros (2,2);
    CVaRout = zeros (2,2);
    CVaRinN = zeros (2,2);
    sort_loss_portf_out_matrix = zeros (100000, 2);
    sort_loss_portf_norm_out_matrix = zeros (100000, 2);
    % Compute VaR and CVaR (non-Normal and Normal) for 100000 scenarios
    for(portN = 1:2)
       % Non-Normal:
       loss_portf_i = Losses_out*x0{portN}; % portf loss for each scenario
       sort_loss_portf_i = sort(loss_portf_i); % sort
       sort_loss_portf_out_matrix (:,portN) = sort_loss_portf_i; % store the sorted losses

       % Normal:
       loss_porft_i_norm_mu = mu_l'*x0{portN}; % using mean losses of each asset to calculate portf losses
       disp (['The mean loss of the portfolio ', num2str(portN), ' is $', num2str(loss_porft_i_norm_mu)])
       loss_porft_i_norm_var = sqrt(x0{portN}'*var_l*x0{portN});% using cov to calculate portf standard derivation
       norm_loss_porft_i = normrnd(loss_porft_i_norm_mu,loss_porft_i_norm_var,Nout,1); % using mean and standard derivation to generate normally distributed random number
       sort_norm_loss_porft_i = sort(norm_loss_porft_i); % sort
       sort_loss_portf_norm_out_matrix(:,portN) = sort_norm_loss_porft_i;% store the sorted losses

       for(q=1:length(alphas))
            alf = alphas(q);
            % -------- Insert your code here -------- %
            VaRout(portN,q)  = sort_loss_portf_i(ceil(Nout*alf)); % find Var
            VaRinN(portN,q)  = sort_norm_loss_porft_i(ceil(Nout*alf)); % find Var of normal case
            CVaRout(portN,q) = (1/(Nout*(1-alf)))*((ceil(Nout*alf)-Nout*alf) * VaRout(portN,q) + sum(sort_loss_portf_i(ceil(Nout*alf)+1:Nout))); % find CVar 
            CVaRinN(portN,q) = (1/(Nout*(1-alf)))*((ceil(Nout*alf)-Nout*alf) * VaRinN(portN,q) + sum(sort_norm_loss_porft_i(ceil(Nout*alf)+1:Nout))); % find CVar Normal case
            % -------- Insert your code here -------- %        
       end
    end

    % Perform 100 trials
    N_trials = 100;
    % Initalize variables to store results
    loss_portf_MC1_matrix = zeros (5000, 100, 2);
    loss_portf_norm_MC1_matrix = zeros (5000, 100, 2);
    loss_portf_MC2_matrix = zeros (5000, 100, 2);
    loss_portf_norm_MC2_matrix = zeros (5000, 100, 2);

    for(tr=1:N_trials)

        % Monte Carlo approximation 1
        Losses_inMC1 = zeros(K,Nin); % Temp for store results of MC1
        % -------- Insert your code here -------- %  
        for s = 1:ceil(Nin/Ns) % systemic scenarios
            y_sim_2 = sqrt_rho * (normrnd(0,1,1,size(rho,1)))'; % systemic scenarios (corelated)
            by_sim_2 = beta.* y_sim_2(driver); % expanded to counterparties
            loss_s_2 = zeros(K,Ns);
            % -------- Insert your code here -------- %
            for si = 1:Ns % idiosyncratic scenarios for each systemic
                % -------- Insert your code here -------- %
                sim_prob_2 = by_sim_2 + sqrt(1-beta.^2).*normrnd(0,1,K,1); % idiosyncratic scenarios
                % Identify the interval (please see report for details)
                sim_prob_dup_2 = repmat(sim_prob_2,1,size(CS_Bdry,2)); 
                sim_diff_2 = -sim_prob_dup_2 + CS_Bdry;
                sim_bool_2 = (sim_diff_2) >= 0;   
                sim_bool_sum_2 = sum(sim_bool_2,2);
                sim_interval_1_2 = C - sim_bool_sum_2; % found the interval
                loss_s_temp_2 = zeros(K,1);
                for i = 1:size(CS_Bdry,1)
                    loss_s_temp_2(i) = exposure(i,sim_interval_1_2(i)); % using interval to find exposure losses
                end
                loss_s_2 (:,si) = loss_s_temp_2;% 100*5 
            end
            Losses_inMC1 (:, 1+(s-1)*5:s*5) = loss_s_2;
        end
        Losses_inMC1 = Losses_inMC1';
        % Calculated losses for MC1 approximation (5000 x 100)
        % Losses_inMC1

        % Monte Carlo approximation 2

        % -------- Insert your code here -------- %
        loss_s_3 = zeros(K, Nin);
        for s = 1:Nin % systemic scenarios (1 idiosyncratic scenario for each systemic)
            % -------- Insert your code here -------- %
            y_sim_3 = sqrt_rho * (normrnd(0,1,1,size(rho,1)))'; % systemic scenarios (corelated)
            by_sim_3 = beta.* y_sim_3(driver); % expanded to counterparties
            % Identify the interval (please see report for details)       
            sim_prob_3 = by_sim_3 + sqrt(1-beta.^2).*normrnd(0,1,K,1);
            sim_prob_dup_3 = repmat(sim_prob_3,1,size(CS_Bdry,2));
            sim_diff_3 = -sim_prob_dup_3 + CS_Bdry;
            sim_bool_3 = (sim_diff_3) >= 0;   
            sim_bool_sum_3 = sum(sim_bool_3,2);
            sim_interval_1_3 = C - sim_bool_sum_3; % found the interval
            loss_s_temp_3 = zeros(K,1);
            for i = 1:size(CS_Bdry,1)
                loss_s_temp_3(i) = exposure(i,sim_interval_1_3(i)); % using interval to find exposure losses
            end
            loss_s_3(:,s) = loss_s_temp_3; % 100*5000
        end
        Losses_inMC2 = loss_s_3';
        % Calculated losses for MC2 approximation (5000 x 100)
        % Losses_inMC2

        % Compute VaR and CVaR
        for(portN = 1:2)
            portf_loss_inMC1 = Losses_inMC1*x0{portN}; % find the portf loss of MC1
            portf_loss_inMC2 = Losses_inMC2*x0{portN};% find the portf loss of MC2
            mu_MCl = mean(Losses_inMC1)'; % find the mean of portf loss of MC1
            var_MCl = cov(Losses_inMC1); % find the cov of portf loss of MC1
            mu_MC2 = mean(Losses_inMC2)'; % find the mean of portf loss of MC2
            var_MC2 = cov(Losses_inMC2); % find the mean cov portf loss of MC2

           % Non-Normal: MC1
           sort_loss_inMC1_i = sort(portf_loss_inMC1); % sort MC1 Non-Normal 
           loss_portf_MC1_matrix (:,tr,portN) = sort_loss_inMC1_i; % store the sorted results

           % Normal: MC1
           loss_porft_MC1_i_norm_mu = mu_MCl'*x0{portN}; % using mean losses of each asset to calculate portf losses MC1
           loss_porft_MC1_i_norm_var = sqrt(x0{portN}'*var_MCl*x0{portN}); % using cov to calculate portf losses MC1
           norm_loss_porft_MC1_i = normrnd(loss_porft_MC1_i_norm_mu,loss_porft_MC1_i_norm_var,Nin,1); % using mean and standard derivation to generate normally distributed random number
           sort_norm_loss_porft_MC1_i = sort(norm_loss_porft_MC1_i); %sort
           loss_portf_norm_MC1_matrix (:,tr,portN) = sort_norm_loss_porft_MC1_i; % store the sorted results

           % Non-Normal: MC2
           sort_loss_inMC2_i = sort(portf_loss_inMC2); % sort MC2 Non-Normal 
           loss_portf_MC2_matrix (:,tr,portN) = sort_loss_inMC2_i; % store the sorted results
           
           % Normal: MC2
           loss_porft_MC2_i_norm_mu = mu_MC2'*x0{portN}; % using mean losses of each asset to calculate portf losses MC2
           loss_porft_MC2_i_norm_var = sqrt(x0{portN}'*var_MC2*x0{portN});  % using cov to calculate portf losses MC2
           norm_loss_porft_MC2_i = normrnd(loss_porft_MC2_i_norm_mu,loss_porft_MC2_i_norm_var,Nin,1); % using mean and standard derivation to generate normally distributed random number
           sort_norm_loss_porft_MC2_i = sort(norm_loss_porft_MC2_i); %sort
           loss_portf_norm_MC2_matrix (:,tr,portN) = sort_norm_loss_porft_MC2_i; % store the sorted results

            for(q=1:length(alphas))
                alf = alphas(q);
                % -------- Insert your code here -------- %            
                % Compute portfolio loss                    
                % Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
                % Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
                % Compute VaR and CVaR for the current trial
                % Refer the Out-of-Sample Case, same process
                VaRinMC1{portN,q}(tr) = sort_loss_inMC1_i(ceil(Nin*alf));
                VaRinMC2{portN,q}(tr) = sort_loss_inMC2_i(ceil(Nin*alf));
                VaRinN1{portN,q}(tr) = sort_norm_loss_porft_MC1_i(ceil(Nin*alf));
                VaRinN2{portN,q}(tr) = sort_norm_loss_porft_MC2_i(ceil(Nin*alf));
                CVaRinMC1{portN,q}(tr) = (1/(Nin*(1-alf)))*((ceil(Nin*alf)-Nin*alf) * VaRinMC1{portN,q}(tr) + sum(sort_loss_inMC1_i(ceil(Nin*alf)+1:Nin)));
                CVaRinMC2{portN,q}(tr) = (1/(Nin*(1-alf)))*((ceil(Nin*alf)-Nin*alf) * VaRinMC2{portN,q}(tr) + sum(sort_loss_inMC2_i(ceil(Nin*alf)+1:Nin)));
                CVaRinN1{portN,q}(tr) = (1/(Nin*(1-alf)))*((ceil(Nin*alf)-Nin*alf) * VaRinN1{portN,q}(tr) + sum(sort_norm_loss_porft_MC1_i(ceil(Nin*alf)+1:Nin)));
                CVaRinN2{portN,q}(tr) = (1/(Nin*(1-alf)))*((ceil(Nin*alf)-Nin*alf) * VaRinN2{portN,q}(tr) + sum(sort_norm_loss_porft_MC2_i(ceil(Nin*alf)+1:Nin)));
                % -------- Insert your code here -------- %
            end
        end
    end
   
    % find the mean and standard derivation of Monte Carlo Simulations for
    % Sampling Error Analysis and Plot
    % Initialze to store variables
    sort_loss_portf_MC1_matrix = zeros (5000,2);
    MC1_5000_std = zeros (5000,2);
    sort_loss_portf_norm_MC1_matrix = zeros (5000,2);
    MC1_norm_5000_std = zeros (5000,2);
    sort_loss_portf_MC2_matrix = zeros (5000,2);
    MC2_5000_std = zeros (5000,2);
    sort_loss_portf_norm_MC2_matrix = zeros (5000,2);
    MC2_norm_5000_std = zeros (5000,2);
    for portN = 1:2 % for each portf
        sort_loss_portf_MC1_matrix(:,portN) = mean(loss_portf_MC1_matrix(:,:,portN),2); % find mean of MC1 for sorted scenarios (of 100 trials)
        MC1_5000_std(:,portN) = std (loss_portf_MC1_matrix(:,:,portN),0,2); % find std of MC1 for sorted scenarios (of 100 trials)
        sort_loss_portf_norm_MC1_matrix(:,portN) = mean(loss_portf_norm_MC1_matrix(:,:,portN),2); % find mean of Normal MC1 for sorted scenarios (of 100 trials)
        MC1_norm_5000_std(:,portN) = std (loss_portf_norm_MC1_matrix(:,:,portN),0,2); % find std of Normal MC1 for sorted scenarios (of 100 trials)
        sort_loss_portf_MC2_matrix(:,portN) = mean(loss_portf_MC2_matrix(:,:,portN),2); % find mean of MC2 for sorted scenarios (of 100 trials)
        MC2_5000_std(:,portN) = std (loss_portf_MC2_matrix(:,:,portN),0,2); % find std of MC2 for sorted scenarios (of 100 trials)
        sort_loss_portf_norm_MC2_matrix(:,portN) = mean(loss_portf_norm_MC2_matrix(:,:,portN),2); % find mean of normal MC2 for sorted scenarios (of 100 trials)
        MC2_norm_5000_std(:,portN) = std (loss_portf_norm_MC2_matrix(:,:,portN),0,2); % find std of normal MC2 for sorted scenarios (of 100 trials)
    end
    MC_1_std_mean = mean(MC1_5000_std); % The 5000 mean of std MC1
    MC_1_norm_std_mean = mean(MC1_norm_5000_std);  % The 5000 mean of std MC1 Normal
    MC_2_std_mean = mean(MC2_5000_std);  % The 5000 mean of std MC2
    MC_2_norm_std_mean = mean(MC2_norm_5000_std);  % The 5000 mean of std MC2 Normal
    disp (['The Mean Standard Deviation of MC1 in-sample for Portf 1 is ', num2str(MC_1_std_mean(1))])
    disp (['The Mean Standard Deviation of MC1 in-sample for Portf 2 is ', num2str(MC_1_std_mean(2))])
    disp (['The Mean Standard Deviation of MC2 in-sample for Portf 1 is ', num2str(MC_2_std_mean(1))])
    disp (['The Mean Standard Deviation of MC2 in-sample for Portf 2 is ', num2str(MC_2_std_mean(2))])
    
    % Display portfolio VaR and CVaR
    for(portN = 1:2)
    fprintf('\nPortfolio %d:\n\n', portN)    
     for(q=1:length(alphas))
        alf = alphas(q);
        fprintf('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRout(portN,q), 100*alf, CVaRout(portN,q))
        fprintf('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC1{portN,q}), 100*alf, mean(CVaRinMC1{portN,q}))
        fprintf('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC2{portN,q}), 100*alf, mean(CVaRinMC2{portN,q}))
        fprintf(' In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRinN(portN,q), 100*alf, CVaRinN(portN,q))
        fprintf(' In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinN1{portN,q}), 100*alf, mean(CVaRinN1{portN,q}))
        fprintf(' In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n\n', 100*alf, mean(VaRinN2{portN,q}), 100*alf, mean(CVaRinN2{portN,q}))
     end
    end

    % Plot results
    for(portN = 1:2)
        % for each portf
        figure
        % True vs MC1 and MC2
        histogram(sort_loss_portf_out_matrix (:,portN),150,'Normalization','Probability');
        hold on
        histogram(sort_loss_portf_MC1_matrix (:,portN),150,'Normalization','Probability');
        hold on
        histogram(sort_loss_portf_MC2_matrix (:,portN),150,'Normalization','Probability');
        hold on
        temp_Var99 = sort_loss_portf_out_matrix (ceil(.99*Nout), portN); 
        line([temp_Var99, temp_Var99],[0 0.05], 'Color', 'r','LineWidth',1);
        hold on
        temp_Var99 = sort_loss_portf_MC1_matrix (ceil(.99*Nin), portN); 
        line([temp_Var99 temp_Var99],[0 0.05], 'Color', 'g','LineWidth',1);
        hold on
        temp_Var99 = sort_loss_portf_MC2_matrix (ceil(.99*Nin), portN); 
        line([temp_Var99 temp_Var99],[0 0.05], 'Color', 'y','LineWidth',1);        
        title (['True Distribution Versus Monte Carlo Simulations for Portf ', num2str(portN)])
        xlabel ('Loss')
        ylabel ('Probability')
        legend ('True Distribution', 'MC1 Distribution', 'MC2 Distribution', 'True Distribution Var 99%', 'MC1 Distribution Var 99%', 'MC2 Distribution Var 99%')
        hold off
        % True vs True-Normal
        figure
        histogram(sort_loss_portf_out_matrix (:,portN),150,'Normalization','Probability');
        hold on
        histogram(sort_loss_portf_norm_out_matrix (:,portN),150,'Normalization','Probability');
        hold on
        temp_Var99 = sort_loss_portf_out_matrix (ceil(.99*Nout), portN); 
        line([temp_Var99, temp_Var99],[0 0.05], 'Color', 'r','LineWidth',1);
        hold on
        temp_Var99 = sort_loss_portf_norm_out_matrix (ceil(.99*Nout), portN); 
        line([temp_Var99 temp_Var99],[0 0.05], 'Color', 'g','LineWidth',1);         
        title (['True Distribution Versus True Normal for Portf ', num2str(portN)])
        xlabel ('Loss')
        ylabel ('Probability')
        legend ('True Distribution', 'Out-of-Sample Normal','True Distribution Var 99%', 'Out-of-Sample Normal Var 99%')
        hold off
        % MC-1 vs MC1-Normal (Uncomment to Plot, MC vs MC Normal is unimportant for analysis errors)
%         figure
%         histogram(sort_loss_portf_MC1_matrix (:,portN),150,'Normalization','probability');
%         hold on
%         histogram(sort_loss_portf_norm_MC1_matrix (:,portN),150,'Normalization','probability');
%         title (['MC1 Distribution Versus MC1 Normal for Portf ', num2str(portN)])
%         xlabel ('Loss')
%         ylabel ('Probability')
%         legend ('MC1', 'MC1 Normal')
%         hold off   
%         % MC-2 vs MC2-Normal
%         figure
%         histogram(sort_loss_portf_MC2_matrix (:,portN),150,'Normalization','probability');
%         hold on
%         histogram(sort_loss_portf_norm_MC2_matrix (:,portN),150,'Normalization','probability');
%         title (['MC2 Distribution Versus MC2 Normal for Portf ', num2str(portN)])
%         xlabel ('Loss')
%         ylabel ('Probability')
%         legend ('MC2', 'MC2 Normal')
%         hold off           
    end