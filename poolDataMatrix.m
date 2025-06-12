function Theta = poolDataMatrix(x, nvars, polyorder)
    % Function to create a library of candidate functions
    % INPUTS:
    % x - data matrix (observations x variables)
    % nvars - number of variables in x
    % polyorder - maximum polynomial order
    
    % Number of observations
    n = size(x, 1);
    
    % Initialize Theta with a column of ones for the constant term
    Theta = ones(n, 1);
    
    % Include the original variables (identity terms)
    for i = 1:nvars
        Theta = [Theta, x(:, i)];
    end
    
    % Include polynomial terms
    if polyorder >= 2
        for i = 1:nvars
            for j = i:nvars
                Theta = [Theta, x(:, i) .* x(:, j)];
            end
        end
    end
    
    if polyorder >= 3
        for i = 1:nvars
            for j = i:nvars
                for k = j:nvars
                    Theta = [Theta, x(:, i) .* x(:, j) .* x(:, k)];
                end
            end
        end
    end
    
    % Continue for higher polynomial orders if needed
    % Add more nested loops for polyorder >= 4, >= 5, etc.
    
    % Output the candidate function matrix
    Theta = Theta;
end
