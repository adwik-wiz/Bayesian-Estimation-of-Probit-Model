%% Classical estimation of Probit Model

% Read the table
data = readtable('hmda.xlsx', 'Sheet', 'data');

% variables
varNames = data.Properties.VariableNames;
covariateNames = setdiff(varNames, {'deny'});
X = table2array(data(:, covariateNames));
y = data.deny;

X_design = [ones(size(X,1),1) X];
[n,k] = size(X_design);

% Run the Probit Model
[b, ~, stats] = glmfit(X, y, 'binomial', 'link', 'probit');

% Store classical estimates in b_classical so we can reference later
b_classical = b;

% Display the maximum likelihood estimates and standard errors
fprintf('Classical Probit Model Estimation Results:\n');
fprintf('-------------------------------------------\n');
fprintf('%-20s %-12s %-12s\n', 'Parameter', 'Estimate', 'Std. Error');
fprintf('%-20s %-12.4f %-12.4f\n', 'Intercept', b(1), stats.se(1));
for i = 2:length(b)
    fprintf('%-20s %-12.4f %-12.4f\n', covariateNames{i-1}, b(i), stats.se(i));
end

%% Bayesian binary probit model

% MCMC settings
nIter = 12500;
burn = 2500;
nStore = 10000;
beta_store = zeros(nStore, k);

% Prior: beta ~ N(0, 100*I)
beta0 = zeros(k,1);
V0 = 100*eye(k);
V0_inv = (1/100)*eye(k);

% Initialize beta and latent variable z
beta = zeros(k,1);
z = zeros(n,1);

% MCMC sampling
for iter = 1:nIter
    
    % (1) Sample latent variables z_i
    for i = 1:n
        mu_i = X_design(i,:) * beta;
        if y(i) == 1
            a = 0;      % Truncate from 0 to +∞ if y=1
            bnd = Inf;
        else
            a = -Inf;   % Truncate from -∞ to 0 if y=0
            bnd = 0;
        end
        
        z(i) = truncated_normal_sample(mu_i, 1, a, bnd);
    end
    
    % Sample beta from its full conditional
    % beta ~ N(m_post, V_post)
    V_post = inv(X_design' * X_design + V0_inv);
    m_post = V_post * (X_design' * z + V0_inv * beta0);
    beta = m_post + chol(V_post, 'lower') * randn(k,1);
    
    % Store draws after burn-in
    if iter > burn
        beta_store(iter - burn, :) = beta';
    end
end

% Compute posterior summary statistics
posterior_mean = mean(beta_store)';
posterior_std = std(beta_store)';

%% 4. Display Combined Results

fprintf('\n\nCombined Results: ML Estimates and Bayesian Posterior Summaries\n');
fprintf('-----------------------------------------------------------------------\n');
fprintf('%-15s %-12s %-12s %-12s %-12s\n', 'Parameter', 'ML_Est', 'ML_SE', 'Bayes_Mean', 'Bayes_SD');
fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
    'Intercept', b_classical(1), stats.se(1), posterior_mean(1), posterior_std(1));

for j = 2:k
    fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        covariateNames{j-1}, b_classical(j), stats.se(j), ...
        posterior_mean(j), posterior_std(j));
end

%% 5. Trace Plot of MCMC Draws for All Parameters
figure;
nRows = ceil(sqrt(k));
nCols = ceil(k / nRows);
for j = 1:k
    subplot(nRows, nCols, j);
    plot(beta_store(:,j));
    title(sprintf('Param %d', j));
    xlabel('Iteration');
    ylabel('Value');
end
sgtitle('Trace Plots for MCMC Draws of Probit Model Parameters');


%% --- Helper Function: Truncated Normal Sampler ---
function x = truncated_normal_sample(mu, sigma, a, b)
% Samples one draw from a N(mu, sigma^2) distribution truncated to [a, b].
% Simple rejection sampler for moderate truncation.
    accepted = false;
    while ~accepted
        x_candidate = mu + sigma*randn;
        if (x_candidate >= a) && (x_candidate <= b)
            x = x_candidate;
            accepted = true;
        end
    end
end
