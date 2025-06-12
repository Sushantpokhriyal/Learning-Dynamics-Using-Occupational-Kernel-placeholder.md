tStart = tic;

% --- Time discretization ---
mu = .1;
h = 1/100;
tspan = 0:h:40;
 %% System parameters
    epsilon = 1e-6;  % Regularization parameter
    
    %% Define the Thomas system
    Thomas_dynamics = @(t,X) [
        sin(X(2)) - 0.2*X(1);
        sin(X(3)) - 0.2*X(2);
        sin(X(1)) - 0.2*X(3)
    ];
    
    %% Generate training data
    x0 = [1.0; -1.0; 0.0];  % Initial condition
    options = odeset('RelTol',1e-6,'AbsTol',1e-9);
    [t, X] = ode45(Thomas_dynamics, tspan, x0, options);
    %X = X';
N = size(X, 1);
d = size(X, 2);  % 3D

% --- Plot trajectory ---
figure;
plot3(X(:, 1), X(:, 2), X(:, 3), 'b--', 'LineWidth', 1.5);
title('Thomas Attractor Trajectory');
xlabel('x'); ylabel('y'); zlabel('z'); grid on;
axis equal;

% --- Polynomial basis functions up to order 2 ---
polyorder = 2;
B = poolDataMatrix(X, d, polyorder);  % user-defined
C = B(:, 2:end);  % remove constant term
M_poly = size(C, 2);

% --- Trigonometric scalar functions ---
trig_scalars = [ ...
    sin(X(:,1)), sin(X(:,2)), sin(X(:,3)), ...
    cos(X(:,1)), cos(X(:,2)), cos(X(:,3)) ];
M_trig = size(trig_scalars, 2);

% --- Combine scalar features ---
all_scalars = [C, trig_scalars];
M_total = size(all_scalars, 2);

% --- Promote scalar to vector-valued basis ---
Y_all = cell(1, d * M_total);
for i = 1:M_total
    for j = 1:d
        tmp = zeros(N, d);
        tmp(:, j) = all_scalars(:, i);
        Y_all{d*(i-1) + j} = tmp;
    end
end

% --- Kernel gradient difference: ∇K(X, X0) - ∇K(X, XT) ---
X1 = X(1, :);
Xend = X(end, :);
K_Xj_X1 = exp(-sum((X - X1).^2, 2) / (2 * mu^2));
K_Xj_Xend = exp(-sum((X - Xend).^2, 2) / (2 * mu^2));
grad_K_diff = ((K_Xj_X1 .* (X - X1)) - (K_Xj_Xend .* (X - Xend))) / mu^2;

% --- Compute Y vector via inner products ---
Y = zeros(length(Y_all), 1);
for i = 1:length(Y_all)
    integrand = sum(grad_K_diff .* Y_all{i}, 2);
    Y(i) = trapz(t, integrand);
end

% --- Kernel Hessian ---
DX = permute(X, [1 3 2]) - permute(X, [3 1 2]);  % (i,j,:) = X(i,:) - X(j,:)
norm_sq = sum(DX.^2, 3);
Kval = exp(-norm_sq / (2 * mu^2));

H = zeros(N, N, d, d);
for i = 1:d
    for j = 1:d
        delta = (i == j);
        H(:, :, i, j) = (delta / mu^2 - (DX(:,:,i) .* DX(:,:,j)) / mu^4) .* Kval;
    end
end

% --- Simpson weights ---
SIMP = ones(N, 1);
SIMP(2:2:end-1) = 4;
SIMP(3:2:end-2) = 2;

% --- Gram matrix computation ---
Mfull = length(Y_all);
grammat = zeros(Mfull, Mfull);

for k = 1:Mfull
    Yk = Y_all{k};
    for l = k:Mfull
        Yl = Y_all{l};
        f_matrix = zeros(N, N);
        for i = 1:d
            for j = 1:d
                f_matrix = f_matrix + (Yk(:, i) * Yl(:, j)') .* squeeze(H(:, :, i, j));
            end
        end
        grammat(k, l) = h^2 / 9 * (SIMP' * f_matrix * SIMP);
        grammat(l, k) = grammat(k, l);
    end
end

% --- Solve regularized least squares problem ---
lambda = 1e-2;
ThetaMat = (grammat + lambda * eye(size(grammat))) \ Y;

% --- Threshold to retain top 10 terms ---
[~, idx] = maxk(abs(ThetaMat), 10);
FinalTheta = zeros(size(ThetaMat));
FinalTheta(idx) = ThetaMat(idx);
disp(FinalTheta);

% --- Reconstruct vector field ---
f_sum = zeros(size(X));
for k = 1:length(Y_all)
    f_sum = f_sum + FinalTheta(k) * Y_all{k};
end

% --- True vector field from Thomas dynamics ---
true_vec = zeros(size(X));
for i = 1:N
    true_vec(i, :) = Thomas_dynamics([], X(i, :))';
end

% --- Mean squared error ---
mse = mean(vecnorm(f_sum - true_vec, 2, 2).^2);
fprintf('Mean Squared Error: %.6e\n', mse);

% --- Quiver3 plot of true vs. recovered ---
figure;
quiver3(X(:,1), X(:,2), X(:,3), f_sum(:,1), f_sum(:,2), f_sum(:,3), 'r');
hold on;
quiver3(X(:,1), X(:,2), X(:,3), true_vec(:,1), true_vec(:,2), true_vec(:,3), 'b--');
legend('Approximated field', 'True field');
title('Thomas Attractor: Vector Field Comparison');
axis equal;
grid on;

toc(tStart);
