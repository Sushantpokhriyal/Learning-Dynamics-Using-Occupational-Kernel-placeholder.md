tStart = tic;

% --- Time discretization ---
mu = 0.1;
h = 1/100;
tspan = 0:h:20;

% --- Initial conditions for 2D dynamics ---
initial_conditions = [rand(), rand()];

% --- 2D Rotational Dynamics: dx/dt = -y, dy/dt = x ---
Rotational_dynamics = @(t, X) [-X(2); X(1)];

% --- Solve the ODE ---
[t, X] = ode45(Rotational_dynamics, tspan, initial_conditions);  % X is [N x 2]
N = size(X, 1);  % number of time samples
d = size(X, 2);  % number of variables (2)

% --- Plot 2D trajectory ---
figure;
plot(X(:, 1), X(:, 2));
title('2D Rotational Dynamics');
xlabel('x'); ylabel('y'); grid on;

% --- Construct polynomial basis functions up to order 2 ---
polyorder = 2;
B = poolDataMatrix(X, d, polyorder);  % user-defined helper
C = B(:, 2:end);  % remove constant column
M = size(C, 2);

% --- Construct directional basis Y_all: 2 per basis function ---
Y_all = cell(1, 2 * M);
for i = 1:M
    Y_all{2*(i-1)+1} = [C(:, i), zeros(N, 1)];  % x-dir
    Y_all{2*(i-1)+2} = [zeros(N, 1), C(:, i)];  % y-dir
end

% --- Kernel gradient difference: ∇K(X, X0) - ∇K(X, XT) ---
X1 = X(1, :);
Xend = X(end, :);

K_Xj_X1   = exp(-sum((X - X1).^2, 2) / (2 * mu^2));
K_Xj_Xend = exp(-sum((X - Xend).^2, 2) / (2 * mu^2));

grad_K_diff = ((K_Xj_X1 .* (X - X1)) - (K_Xj_Xend .* (X - Xend))) / mu^2;

% --- Compute Y via trapezoidal integral of ∇K_diff · Y_i ---
Y = zeros(length(Y_all), 1);
for i = 1:length(Y_all)
    integrand = sum(grad_K_diff .* Y_all{i}, 2);  % ⟨∇K_diff(x), Y_i(x)⟩
    Y(i) = trapz(t, integrand);
end

% --- Kernel Hessian tensor H(x, y) of shape N x N x d x d ---
DX = permute(X, [1 3 2]) - permute(X, [3 1 2]);  % DX(i,j,:) = X(i,:) - X(j,:)
norm_sq = sum(DX.^2, 3);
Kval = exp(-norm_sq / (2 * mu^2));

H = zeros(N, N, d, d);
for i = 1:d
    for j = 1:d
        delta = (i == j);
        H(:, :, i, j) = (delta / mu^2 - (DX(:, :, i) .* DX(:, :, j)) / mu^4) .* Kval;
    end
end

% --- Simpson’s rule weights for higher accuracy ---
SIMP = ones(N, 1);
SIMP(2:2:end-1) = 4;
SIMP(3:2:end-2) = 2;
SIMP = SIMP(:);

% --- Compute gramm matrix: double contraction over H and basis ---
Mfull = length(Y_all);
grammat = zeros(Mfull, Mfull);

for k = 1:Mfull
    Yk = Y_all{k};  % [N x d]
    for l = k:Mfull
        Yl = Y_all{l};
        f_matrix = zeros(N, N);
        for i = 1:d
            for j = 1:d
                f_matrix = f_matrix + (Yk(:, i) * Yl(:, j)') .* squeeze(H(:, :, i, j));
            end
        end
        grammat(k, l) = h^2 / 9 * (SIMP' * f_matrix * SIMP);
        grammat(l, k) = grammat(k, l);  % symmetry
    end
end

% --- Solve for sparse representation using Tikhonov regularization ---
lambda = 1e-2;
ThetaMat = (grammat + lambda * eye(size(grammat))) \ Y;

% --- Threshold: retain top 10 coefficients by magnitude ---
[~, idx] = maxk(abs(ThetaMat), 10);
FinalTheta = zeros(size(ThetaMat));
FinalTheta(idx) = ThetaMat(idx);
disp(FinalTheta);

% --- Reconstruct approximated vector field ---
f_sum = zeros(size(X));
for k = 1:length(Y_all)
    f_sum = f_sum + FinalTheta(k) * Y_all{k};
end

% --- True dynamics from definition: [-y, x] for 2D rotation ---
true_vec = [-X(:, 2), X(:, 1)];

% --- Compute model error ---
mse = mean(vecnorm(f_sum - true_vec, 2, 2).^2);
fprintf('Mean Squared Error: %.6e\n', mse);

% --- Optional: Visualize field recovery ---
figure;
quiver(X(:,1), X(:,2), f_sum(:,1), f_sum(:,2), 'r--');
hold on;
quiver(X(:,1), X(:,2), true_vec(:,1), true_vec(:,2), 'b');
legend('Approximated field', 'True field');
title('Vector Field Comparison');
axis equal;
grid on;

toc(tStart);

