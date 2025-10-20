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
figure;
set(gca, 'FontSize', 14);
hold on;
for sim = 1:num_simulations
    plot(1:num_iter, all_cost_history(:, sim), '-', 'Color', [0.7 0.7 0.7]);
end
plot(1:num_iter, avg_cost_history, 'k-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 16);
ylabel('Cost', 'FontSize', 16);
title('Cost History for Each Simulation (Adam Optimiser)', 'FontSize', 18);
legend(sprintf('Average (%d Simulations)', num_simulations), 'Location', 'Best', 'FontSize', 14);
grid on;

% ---------------------- SAVE CSV ---------------------- %
data_to_save = [1:num_iter; avg_cost_history'];
csv_filename = 'cost_history_data_adam.csv';
csvwrite(csv_filename, data_to_save);
fprintf('\nData has been saved to %s.\n', csv_filename);

fprintf('Cost after %d iterations (average simulation): %.5e\n', num_iter, avg_cost_history(end));

