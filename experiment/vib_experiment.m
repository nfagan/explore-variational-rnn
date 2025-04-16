%%

W_mu = ri( 2, 1 );
W_cov = ri( 3, 1 );
W_p = ri( 2, 2 );

x = ri( 1, 1 );

mu = W_mu * x;
sigma = genmvn_sigma( W_cov * x );

z = zeros( 1e3, 2 );
q_y = zeros( size(z, 1), 2 );

for i = 1:1e3

z(i, :) = mvnrnd( mu, sigma );
q_y(i, :) = softmax( W_p * z(i, :)' );

end

%

clf;
scatter( z(:, 1), z(:, 2) );
axis( 'square' );
ylim( [-4, 4] );
xlim( [-4, 4] );

%%

function y = softmax(x, varargin)
y = exp(x) ./ sum(exp(x), varargin{:});
end

function y = softplus(x)
y = log( 1 + exp(x) );
end

function sigma = genmvn_sigma(cov_proj)
sigma = zeros( 2 );
sigma(1, 1) = cov_proj(1);
sigma(2, 2) = cov_proj(2);
sigma(2, 1) = cov_proj(3);
d = softplus( diag(sigma) );
sigma(1, 1) = d(1);
sigma(2, 2) = d(2);
sigma = sigma * sigma';
end

function w = ri(m, n)
w = rand( m, n ) * 2 - 1;
end