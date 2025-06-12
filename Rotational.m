
tStart = tic;  % Start a timer to measure total elapsed time

% --- time discretization ---
mu = 10;
h = 1/100;               % Time step size
tspan = 0:h:20;          % Time vector from 0 to 20 with step size h

% --- Initial conditions for ODE ---
initial_conditions = [rand(), rand()];  % Starting state of Lorenz system

% --- Define the Lorenz system as a vector field X' = f(X) ---
Rotational_dynamics = @(t, X) [
     - X(2);
    X(1)
];

% -- Numerically integrate the system using ode45 (adaptive Runge-Kutta) ---
[t, X] = ode45(Rotational_dynamics, tspan, initial_conditions);

% --- Normalize the trajectory for numerical stability (zero mean, unit variance) ---
%X = (X - mean(X)) ./ std(X);  % Each column is normalized



% --- Plot the attractor in 3D space ---
figure;
plot(X(:, 1), X(:, 2));
title('Rotional Dynamics');


% --- Construct a library of candidate functions (polynomial basis functions) ---
nvars = 2;                 % Number of state variables: x, y, z
polyorder = 2;             % Degree of polynomials used in feature space
B = poolDataMatrix(X, nvars, polyorder);  % Generate all monomials up to 3rd order
C = B(:, 2:end);           % Remove the constant term (column 1 is all ones)

% --- Construct basis-aligned matrix set Y_all (to associate each basis function with one coordinate) ---
M = size(C, 2);            % Number of polynomial basis functions (excluding constant)
Y_all = cell(1, 2 * M);    % Allocate cell array to hold all directional basis matrices

% % Fill Y_all with directional embeddings of each basis function
 for i = 1:M
      Y_all{2*(i-1)+1} = [C(:,i), zeros(size(C,1), 1)];         % X-component only
      %Y_all{2*(i-1)+2} = [zeros(size(C,1), 1), C(:,i), zeros(size(C,1), 1)];  % Y-component
    Y_all{2*(i-1)+2} = [zeros(size(C,1), 1), C(:,i)];         % Z-component
 end
% 
% % --- Compute kernel gradient difference ∇_x K(x, x0) - ∇_x K(x, xT) ---
 X1 = X(1,:);               % Initial point
 Xend = X(end,:);           % Final point
 K_Xj_Xend = exp(-sum((X - Xend).^2, 2) / (2 * mu^2));  % Gaussian RBF K(x_j, x_end)
 K_Xj_X1 = exp(-sum((X - X1).^2, 2) / (2 * mu^2));      % Gaussian RBF K(x_j, x_1)
% 
% % ∇_x K(x,x0) = -K(x,x0) * (x - x0) / mu^2; similarly for xT
 grad_K_diff = ((K_Xj_X1 .* (X - X1)) - (K_Xj_Xend .* (X - Xend))) / mu^2;
% 
% % --- Compute RHS vector Y using trapezoidal integration of inner product between gradient and Y_all basis ---
 Y = zeros(length(Y_all), 1);
 for i = 1:length(Y_all)
     Y_local = Y_all{i};                           % Each directional candidate basis matrix
     integrand = sum(grad_K_diff .* Y_local, 2);   % Inner product ⟨∇K_diff(x), y_i(x)⟩
     Y(i) = trapz(t,integrand);                   % Approximate integral using trapezoidal rule
 end
% 
% % --- Vectorized computation of pairwise kernel matrix and its Hessians ---
 N = size(X, 1);                                   % Number of time samples
 DX = permute(X, [1 3 2]) - permute(X, [3 1 2]);   % DX(i,j,:) = X(i,:) - X(j,:) => shape (N x N x 3)
 norm_sq = sum(DX.^2, 3);                          % Squared distance matrix ||X_i - X_j||^2
 Kval = exp(-norm_sq / (2 * mu^2));                % Gaussian RBF kernel matrix K(X_i, X_j)
 
% % --- Compute mixed second derivatives (∇_x ∇_y K(x, y)) = H_ij ---
  H(i,j,k,l) = (δ_kl / mu^2 - (x_k - y_k)(x_l - y_l)/mu^4) * K(x,y)
 H = zeros(N, N, 3, 3);
 for i = 1:3
     for j = 1:3
         delta_ij = (i == j);  % Kronecker delta
         H(:,:,i,j) = delta_ij / mu^2 - (DX(:,:,i) .* DX(:,:,j)) / mu^4;
    end
 end
H = H .* Kval;  % Scale Hessian components by kernel value element-wise
% 
% % --- Simpson's rule weights (for higher-accuracy integration) ---
 SIMP = [1, 3 + (-1).^(1:N-2), 1];  % 1, 4, 2, 4, ..., 4, 1 pattern SIMP = SIMP(:);  % Column vector
% 
% % --- Compute Grammat matrix: double integral of y_i^T H y_j over time-time ---
 Mfull = length(Y_all);
 grammat = zeros(Mfull, Mfull);
% 
 for k = 1:Mfull
     Yk = Y_all{k};  % (N x 3) basis function k
    for l = k:Mfull
        Yl = Y_all{l};  % (N x 3) basis function l
        f_matrix = zeros(N, N);  % accumulator for integrand matrix

        % Compute double contraction y_i(x)^T H(x,y) y_j(y)
        for i = 1:3
            for j = 1:3
                f_matrix = f_matrix + (Yk(:,i) * Yl(:,j)') .* squeeze(H(:,:,i,j));
            end
        end

        % Apply composite Simpson's rule: ∫∫ yᵢᵗ H yⱼ ≈ h²/9 * SIMPᵗ * f_matrix * SIMP
        grammat(k,l) = h^2 / 9 * (SIMP' * f_matrix * SIMP);
        grammat(l,k) = grammat(k,l);  % exploit symmetry
    end
end

% --- Solve the regularized linear system (ridge regression) ---
lambda = 1e-2;  % regularization parameter to avoid ill-conditioning
ThetaMat = (grammat + lambda * eye(size(grammat))) \ Y;
disp(ThetaMat);  % Print the learned coefficients

% --- Keep top 10 terms (based on magnitude) ---
[~, idx] = maxk(abs(ThetaMat), 10);
FinalTheta = zeros(size(ThetaMat));
FinalTheta(idx) = ThetaMat(idx);
disp(FinalTheta);  % Sparse model

% --- Reconstruct learned vector field from selected terms ---
f_sum = zeros(size(X));  % N x 3
for k = 1:length(Y_all)
    f_sum = f_sum + FinalTheta(k) * Y_all{k};  % Linear combination of directional basis
end


% --- Compute true Lorenz vector field from the trajectory (ground truth) ---
original_value = [
    sigma * (X(:,2) - X(:,1)), ...
    rho * X(:,1) - X(:,1).*X(:,3) - X(:,2), ...
    X(:,1).*X(:,2) - beta * X(:,3)
];

% --- Frobenius norm of the modeling error ---
error = norm(original_value - f_sum, 'fro');  % Sum-of-squares of all entries
%disp(['Error: ', num2str(error)]);
disp(error);

% --- End timer and report time ---
elapsed_time = toc(tStart);  
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
