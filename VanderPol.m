%%
clc; clear;

% --- Parameters and time discretization ---
mu = 0.5;               % kernel width
eps = 4;             % Van der Pol nonlinearity parameter
h = 0.01;              % time step
tspan = 0:h:40;        % simulation time

% --- Initial condition and ODE definition ---
initial_conditions = [2; 0];
VanDerPol = @(t, X) [X(2); eps*(1 - X(1)^2)*X(2) - X(1)];

% --- Solve the system using ode45 ---
[t, X] = ode45(VanDerPol, tspan, initial_conditions);
X = real(X);   % remove any tiny imaginary parts
N = size(X,1);
d = size(X,2); % should be 2

% --- Plot trajectory ---
figure;
plot(X(:,1), X(:,2));
title('Van der Pol Oscillator');
xlabel('x'); ylabel('y'); grid on;

% --- Construct polynomial library (exclude constant) ---
polyorder = 3;
B = poolDataMatrix(X, d, polyorder); % user-defined function
C = B(:, 2:end);
M = size(C, 2);

% --- Construct directional basis Y_all ---
Y_all = cell(1, 2*M);
for i = 1:M
    Y_all{2*(i-1)+1} = [C(:,i), zeros(N, 1)];  % x-dir
    Y_all{2*(i-1)+2} = [zeros(N, 1), C(:,i)];  % y-dir
end

% --- Kernel gradient difference ---
X1 = X(1,:);
Xend = X(end,:);
K1 = exp(-sum((X - X1).^2, 2) / (2*mu^2));
Kend = exp(-sum((X - Xend).^2, 2) / (2*mu^2));
grad_diff = ((K1 .* (X - X1)) - (Kend .* (X - Xend))) / mu^2;

% --- Compute Y vector ---
Y = zeros(length(Y_all), 1);
for i = 1:length(Y_all)
    integrand = sum(grad_diff .* Y_all{i}, 2);
    Y(i) = trapz(t, integrand);
end

% --- Compute Hessian tensor ---
DX = permute(X, [1 3 2]) - permute(X, [3 1 2]);
norm_sq = sum(DX.^2, 3);
Kmat = exp(-norm_sq / (2*mu^2));

H = zeros(N,N,d,d);
for i = 1:d
    for j = 1:d
        delta = (i==j);
        H(:,:,i,j) = (delta/mu^2 - (DX(:,:,i).*DX(:,:,j))/(mu^4)) .* Kmat;
    end
end

% --- Simpson weights for double integration ---
SIMP = ones(N,1);
SIMP(2:2:end-1) = 4;
SIMP(3:2:end-2) = 2;

% --- Construct Grammatrix ---
grammat = zeros(length(Y_all));
for k = 1:length(Y_all)
    Yk = Y_all{k};
    for l = k:length(Y_all)
        Yl = Y_all{l};
        f_mat = zeros(N,N);
        for i = 1:d
            for j = 1:d
                f_mat = f_mat + (Yk(:,i) * Yl(:,j)').* squeeze(H(:,:,i,j));
            end
        end
        grammat(k,l) = h^2/9 * (SIMP' * f_mat * SIMP);
        grammat(l,k) = grammat(k,l);
    end
end

% --- Solve for coefficients ---
lambda = 1e-3;
Theta = (grammat + lambda * eye(size(grammat))) \ Y;

% --- Select sparse subset ---
[~, idx] = maxk(abs(Theta), 10);
FinalTheta = zeros(size(Theta));
FinalTheta(idx) = Theta(idx);

% --- Reconstruct vector field ---
f_sum = zeros(N,d);
for k = 1:length(Y_all)
    f_sum = f_sum + FinalTheta(k) * Y_all{k};
end

% --- True vector field for comparison ---
true_f = zeros(N, d);
for i = 1:N
    true_f(i,:) = VanDerPol(t(i), X(i,:))';
end

% --- Compute MSE ---
mse = mean(vecnorm(true_f - f_sum, 2, 2).^2);
fprintf('Mean Squared Error: %.6e\n', mse);

% --- Visualize vector field reconstruction ---
figure;
quiver(X(:,1), X(:,2), true_f(:,1), true_f(:,2), 'b');
hold on;
quiver(X(:,1), X(:,2), f_sum(:,1), f_sum(:,2), 'r');
legend('True Field', 'Approximated Field');
title('Van der Pol: Koopman-Kernel Regression');
axis equal;
grid on;