%%

%{

as entropy is an extensive quantity, it will be proportional to the
word-length N (the support of the distribution):
  S(N) \propto S0 * N

Thus, the system is characterized by an energy density S0.

Although entropy grows linearly with N, different systems exhibit different
behavior in sub-linear components: S(N) \propto SO*N + S1*N, where S1 is
the sublinear component. Because it is sublinear, the sub-extensive
component decays to 0 as N -> inf.

I(xf, xp) = 
  D_kl[P(xf, xp) || P(xf)*p(xp)] = 
  <integral> P(xf, xp) * [ log(P(xf, xp) / (P(xf) * P(xp)) ] =
  <integral> P(xf, xp) * [ log(P(xf | xp) / P(xf) ] = 
  = E_P(xf, xp)[ log(P(xf, xp) - log(P(xf)) - log(P(xp)) ]
  = -S[xf, xp] + S[xf] + S[xp], where S is an entropy

lim_T→∞ S(T)/T = S0, so write S(T) = S0*T + S1(T), lim_T→∞ S1(T) = 0

S(F) + S(P) - S(F+P) = 
  S0*F+S1(F) + S0*P+S1(P) - S0*(F+P) - S1(F+P) = 
  S1(F) + S1(P) - S1(F+P)

now take the limit as F -> inf, S1(F) -> 0, S1(F+P) -> 0, I(P) -> S1(P)

%}

%%

%{

a model is an encoding of data. the description length of the model's
parameters is its complexity (nb. is this true?) - and its performance is
the KL divergence between the true distribution P and the model's encoding
Q (nb. is this true?). thus an overparameterized model is one for which
complexity could be reduced while maintaining perfect performance.

y ~ P(y | x) = f(x)
y ~ Q(y | x) = g(x)

integrate_x{ D_kl( P(y | x) || Q(y | x) ) == 0 }

when Q(y | x) is parameterized by a recurrent function f(f(h, x), x), ... 
can multiple applications of f effectively increase the complexity of the
composed function, without increasing its size?

intuitively, we can always increase the model's description length by
adding terms that do not change its output distribution.

%}

%%

%{

increase in expressivity with increasing recurrent depth (T)?

%}

num_its = 1e3;
T = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024];
nc = 10;
% nonlin = @relu;
nonlin = @tanh;
do_time = false;
s = 0.1;

ents = zeros( num_its, numel(T) );
ents2 = zeros( size(ents) );
ncs = zeros( size(ents) );

for idx = 1:size(ents, 1)
  fprintf( '\n %d of %d', idx, size(ents, 1) );
  if ( do_time ), tic; end
  for j = 1:size(ents, 2)
    W = randn( 4 ) * s;
    Wh = randn( size(W) ) * s;
    Wp = randn( nc, size(W, 1) ) * s;
    b = randn( size(W, 1), 1 ) * s;
    bh = randn( size(b) ) * s;
    fs = zeros( 1e4, 1 );
    ps = zeros( numel(fs), nc );
    parfor i = 1:numel(fs)
      x = randn( size(W, 1), 1 ) * s;
      h = zeros( size(x) );
      f = @(x, h) nonlin( W*x + b + Wh*h + bh );
      for t = 1:T(j), h = f( x, h ); end
      z = softmax( Wp * h );
      [~, ind] = max( z );
      fs(i) = ind;
      ps(i, :) = z;
    end
    pp = -ps .* log( ps );
    pp(ps == 0) = 0;
    pp = sum( pp, 2 );
    ents2(idx, j) = mean( pp );
    fs = arrayfun( @(x) sum(fs == x), 1:nc );
    p = fs / sum( fs );
    ent = -p .* log( p );
    ent(p == 0) = 0;
    ents(idx, j) = sum( ent );
    ncs(idx, j) = sum( sum(fs, 1) > 0 );
  end
  if ( do_time ), toc; end
end

%%

clf;
plt = ents;
axs = plots.panels( size(plt, 2) );
for i = 1:numel(axs)
  histogram( axs(i), plt(:, i) );
  title( axs(i), compose("T=%d; mean = %0.3f", T(i), mean(plt(:, i))) );
end
shared_utils.plot.match_ylims( axs );
shared_utils.plot.match_xlims( axs );

%%

clf;
axs = plots.panels( 3 );
plot( axs(1), log2(T), median(ents, 1), 'LineWidth', 2 ); title(axs(1), 'entropy');
plot( axs(2), log2(T), mean(ents2), 'LineWidth', 2 ); title(axs(2), 'entropy of z');
plot( axs(3), log2(T), mean(ncs, 1), 'LineWidth', 2 ); title(axs(3), 'num classes');

%%

%{

biological organisms can make use of variable thinking time. the time they 
spend is proportional to the complexity of the decisions they must make. we 
explore the theory that this variability in thinking time can be explained 
by a speed/accuracy tradeoff in the face of constraints on information 
processing. in particular, we propose that simple, iterative procedures 
represent a good compromise between expected performance and 
representational complexity in the following sense: through iteration, a 
learner can extract more predictive information from a signal using fewer
nonlinear operations and less experience (fewer samples from the signal 
generating process).

%}

%%

function z = softmax(h)
h = h + 1e-12;
z = exp( h ) ./ sum( exp(h) );
end

function y = relu(x)
y = x;
y(y < 0) = 0;
end