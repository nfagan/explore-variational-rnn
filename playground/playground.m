%%  forward

Nx = 2;
Nh = 2;
Nu = 4;
Nc = 2;

Wh = rand( Nh, Nh );
Wx = rand( Nh, Nx );
Wv = rand( Nu * 2, Nh );
Wp = rand( Nc, Nu );

xt = rand( Nx, 1 );
y_t = one_hot( 1, Nc );

h_prev = zeros( Nh, 1 );

ht = tanh( Wh * h_prev + Wx * xt );
mu = Wv * ht;

ms = means( Nu );
vs = vars( Nu );
es = randn( Nu, 1 );

z_t = ms * mu + vs * mu .* es;

logits_t = Wp * z_t;

% classification term
p = y_t * softmax( logits_t );

% kl term
kl = xxx;

L = log( p ) - kl;

%%  backward

dWp = (y_t' - softmax(Wp * z_t)) * z_t';
dWv = xxx;
dWx = xxx;
dWh = xxx;

%%

function V = vars(n)
V = zeros( n, n*2 );
for i = 1:n, V(i, i+n) = 1; end
end

function M = means(n)
%{
z_t = [[1, 0, 0, 0]; [0, 1, 0, 0]] * mu + ...
      [[0, 0, 1, 0]; [0, 0, 0, 1]] * mu .* randn( 2, 1 );
%}
M = zeros( n, n*2 );
for i = 1:n, M(i, i) = 1; end
end

function t = one_hot(i, nc)
t = zeros( 1, nc );
t(i) = 1;
end

function y = softmax(x)
y = exp( x ) ./ sum( exp(x) );
end
