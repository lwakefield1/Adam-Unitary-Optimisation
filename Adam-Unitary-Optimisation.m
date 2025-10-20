% Function to compute cost
computeCost = @(Uref, Utest) norm(Uref - Utest, 'fro')^2;

% Set a fixed seed for reproducibility
rng_seed = 1111;
rng(rng_seed);

% Prompt user for dimensions
d = input("Enter the value of d (dimensions of matrix): ");
n = d;

% Generate random complex matrix and construct Uref
A = randn(n) + 1i * randn(n);
[Q, R] = qr(A);
Uref = Q * diag(diag(R) ./ abs(diag(R)));

% Parameters
num_iter = 50;
num_simulations = input("Enter the number of simulations: ");
learning_rate = 0.15;
beta1 = 0.9;
beta2 = 0.99;
epsilon = 1e-8;

% Initialise arrays
avg_iterations = zeros(1, num_simulations);
best_costs = zeros(1, num_simulations);
best_iterations = zeros(1, num_simulations);
iteration_counts = zeros(1, num_simulations);
best_U_all = cell(1, num_simulations);
all_cost_history = zeros(num_iter, num_simulations);

% ---------------------- MAIN LOOP ---------------------- %
for sim = 1:num_simulations
    P = randn(d);
    m = zeros(size(P));
    v = zeros(size(P));
    cost_function_evaluations = 0;

    best_cost = inf;
    best_U = [];
    best_iter = 0;
    cost_history = zeros(1, num_iter);

    for i = 1:num_iter
        % Compute Utest
        Utest = UC(P);
        cost = computeCost(Uref, Utest);

        % Track best cost
        if cost < best_cost
            best_cost = cost;
            best_U = Utest;
            best_iter = i;
        end

        cost_history(i) = cost;

        % Finite difference gradient
        grad_eps = 1e-6;
        gradient = zeros(size(P));
        for row = 1:d
            for col = 1:d
                P_plus = P;
                P_minus = P;
                P_plus(row, col) = P_plus(row, col) + grad_eps;
                P_minus(row, col) = P_minus(row, col) - grad_eps;

                Utest_plus = UC(P_plus);
                Utest_minus = UC(P_minus);

                gradient(row, col) = ...
                    (computeCost(Uref, Utest_plus) - computeCost(Uref, Utest_minus)) / (2 * grad_eps);
            end
        end

        % Update first and second moment estimates
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * (gradient .^ 2);

        % Bias correction
        m_hat = m / (1 - beta1^i);
        v_hat = v / (1 - beta2^i);

        % Adam update rule
        P = P - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);

        cost_function_evaluations = cost_function_evaluations + 1;

        % Early stopping
        if cost < 0.01
            iteration_counts(sim) = cost_function_evaluations;
            break;
        end
    end

    % Store results
    avg_iterations(sim) = mean(find(cost_history > 0));
    best_costs(sim) = best_cost;
    best_iterations(sim) = best_iter;
    best_U_all{sim} = best_U;
    all_cost_history(:, sim) = cost_history;
end

% ---------------------- ANALYSIS ---------------------- %
avg_cost_history = mean(all_cost_history, 2);

% Display the best result across all simulations
[~, best_sim_idx] = min(best_costs);
fprintf('\nBest simulation: #%d\n', best_sim_idx);
fprintf('Best cost achieved: %.10e\n', best_costs(best_sim_idx));
fprintf('Iteration number (best): %d\n', best_iterations(best_sim_idx));

best_U_overall = best_U_all{best_sim_idx};
disp('Best Unitary Matrix (U_best):');
disp(best_U_overall);

% Save best unitary matrix
save('best_unitary_matrix_adam.mat', 'best_U_overall');
fprintf('Saved best unitary matrix to best_unitary_matrix_adam.mat\n');

% ---------------------- PLOTS ---------------------- %

% 1. Cost history plot
figure('visible', 'on', 'NumberTitle', 'off', 'Name', 'Cost History', 'Position', [200 200 800 600]);
set(gca, 'FontSize', 14);
hold on;
for sim = 1:num_simulations
    plot(1:num_iter, all_cost_history(:, sim), '-', 'Color', [0.7 0.7 0.7]);
end
plot(1:num_iter, avg_cost_history, 'k-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 16);
ylabel('Cost', 'FontSize', 16);
title('Cost History for Each Simulation (Adam Optimiser)', 'FontSize', 18);
legend(sprintf('Average (%d Simulations)', num_simulations), 'Location', 'northeast', 'FontSize', 14);
grid on;
drawnow;

% 2. Percentile plot
figure('visible', 'on', 'NumberTitle', 'off', 'Name', 'Percentile Range', 'Position', [250 250 800 600]);
set(gca, 'FontSize', 14);
percentile_10_history = prctile(all_cost_history, 10, 2);
percentile_50_history = prctile(all_cost_history, 50, 2);
percentile_90_history = prctile(all_cost_history, 90, 2);
best_percentile_history = min(all_cost_history, [], 2);
worst_percentile_history = max(all_cost_history, [], 2);
hold on;
plot(1:num_iter, avg_cost_history, 'k-', 'LineWidth', 2);
plot(1:num_iter, percentile_10_history, 'r--', 'LineWidth', 1.5);
plot(1:num_iter, percentile_50_history, 'c-', 'LineWidth', 1.5);
plot(1:num_iter, percentile_90_history, 'b--', 'LineWidth', 1.5);
plot(1:num_iter, best_percentile_history, 'g-.', 'LineWidth', 1.5);
plot(1:num_iter, worst_percentile_history, 'm-.', 'LineWidth', 1.5);
xlabel('Iteration', 'FontSize', 16);
ylabel('Cost', 'FontSize', 16);
title('Average Cost and Percentile Range (Adam)', 'FontSize', 18);
legend('Average', '10th', 'Median', '90th', 'Best', 'Worst', 'Location', 'northeast', 'FontSize', 14);
grid on;
drawnow;

% 3. Iteration count plot
figure('visible', 'on', 'NumberTitle', 'off', 'Name', 'Iteration Counts', 'Position', [300 300 800 600]);
set(gca, 'FontSize', 14);
bar(iteration_counts);
xlabel('Simulation', 'FontSize', 16);
ylabel('Cost Function Evaluations', 'FontSize', 16);
title('Number of Cost Function Evaluations until Cost < 0.01 (Adam)', 'FontSize', 18);
grid on;
drawnow;


