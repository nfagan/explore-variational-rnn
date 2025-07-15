%%  transformation costs

% express the "cost" of an event in terms of its probability

% pdf of events
p = [0.1, 2, 3, 3]; p = p ./ sum( p );

% sensitivity to computation; try 1 or 8 and observe phi
b = 1;

phis = 2.3; % cost potential for the whole set

% transformation costs are defined by -1/b * log(.). because of the log, 
% lower probability events have higher cost. however, as b is increased, 
% costs are increasingly uniform (pulled toward the cost of the whole set).
% in this sense, as b is decreased, costs are more precise with respect to
% the probabilities of their constituent events.
phi = phis - 1/b .* log( p )

% the cost potential of the whole set is log sum e^(-b * phi)
phis2 = -1/b .* log( sum(exp(-b .* phi)) ); % == phis

%%  gibbs measure & free energy

% Z = e^(-beta * phis) = exp( -b * phis );
Z = @(b, phi) sum(exp(-b .* phi));
p_xs = exp( -b .* phi ) / Z(b, phi); % == p

expected_cost = @(q, phi) sum(q .* phi);
entropy = @(q) -sum(q .* log(q));
free_energy = @(q, b, phi) expected_cost(q, phi) - 1/b * entropy(q);

% to lower free energy, we can either reduce the expected cost or increase 
% the entropy term (increasing entropy in q or reducing beta). an increased
% entropy term can be interpreted as allowing a higher information processing
% budget.
free_energy( p_xs, b, phi )

dkl = @(p, q) sum(p .* log(p./ q));
% "energy baseline": -1/b*log(Z) + KL divergence from optimal p
% as beta increases, the kl term -> 0
free_energy2 = @(q, b, p, phi) -1/b*log(Z(b, phi)) + 1/b*dkl(q, p);

q = rand( size(p) ); q = q ./ sum(q);
free_energy2( p, b, p, phi )
free_energy2( q, b, p, phi )

%%  free energy difference (change in free energy)

p0 = [0.1, 2, 3, 3]; p0 = p0 ./ sum( p0 );
p1 = [2, 0.1, 3, 5]; p1 = p1 ./ sum( p1 );

b = 1;

phis = 2.3; % cost potential for the whole set
phi0 = phis - 1/b .* log( p0 );
phi1 = phis - 1/b .* log( p1 );
U = phi0 - phi1;

% -b*phi1 = -b*phis + log(p1)
% exp(-b * phi1) = p1 * exp(-b*phis)
% p1 = exp(-b .* phi1) / exp(-b * phis)

% because of the direct correspondence between costs and probabilities, a
% change in cost (gain or loss) maps to a change in probability mass from
% p0. so p1p is exactly p1.
p1p = p0 .* exp( b .* U ); % == p1

% negative free energy diff is maximized (i.e., positive free energy diff
% is minimized), when p == p1 == p0(x)*exp(b * U(x)).
% in the face of some gain or loss U, the optimal policy shifts from p0 to
% p. deviation (in the KL-sense) from `p` implies a more negative (larger) 
% free energy diff.
energy_diff = @(p, b) sum(p .* U) - 1/b*dkl(p, p0);
energy_diff(p1, b)

% above, we update p1 and compute phi1. again, costs <=> probs means we can
% change costs and compute probs. however, the total cost, phis, must
% remain fixed.

%%  

%{

(7) p(x) ∝ exp(−β*phi0(x) + β*U(x)) ∝ p0(x) * exp(β*U(x))

"Accordingly, (7) can either be seen as the distribution that maximizes the
entropy [of p] given a constraint on the expectation value of U or as the 
distribution that minimizes the expectation of −U given a constraint on the 
entropy of p."

- to make E[U] large, beta has to be large, minimizing the entropy in p.
  (principle of estimation)
- when the entropy of p is fixed, exp(β * U(x)) is the distribution that
  minimizes -E[U] (or maximizes E[U])
  (principle of bounded rational decision-making)
- in this case, beta is interpreted as the computation required to evaluate 
  gains / losses

%}