
poss_ops = ops;
op_names = fieldnames( poss_ops );

num_ops = 1:10;
num_ops = 1:2:30;
b = [ true, false ];
num_samp = 1e3;

r_ops = cell( numel(num_ops), 1 );
b0_ops = cell( size(r_ops) );
b1_ops = cell( size(r_ops) );
d_ops = cell( size(r_ops) );

% false, false -> [true, false]
% false, true -> [true, false]
% true, false -> [true, false]
% true, true -> [true, false]

i = 1;
for n = num_ops
  os = randi( numel(op_names), n, num_samp );
  os = reshape( op_names(os), size(os) );
  os = cellfun( @(x) poss_ops.(x), os, 'un', 0 );
  rs = zeros( size(os, 2), 2, 2 );
  b0s = false( size(rs) );
  b1s = false( size(rs) );
  d = zeros( size(os, 2), 1 );
  for j = 1:size(os, 2)
    for k = 1:2
      for h = 1:2
        r = evaluate( os(:, j), b(k), b(h) );
        rs(j, k, h) = r;
        b0s(j, k, h) = b(k);
        b1s(j, k, h) = b(h);
      end
    end
    d(j) = bin2dec( char(strjoin(string(reshape(rs(j, :, :), 1, [])))) );
  end
  r_ops{i} = rs;
  b0_ops{i} = b0s;
  b1_ops{i} = b1s;
  d_ops{i} = d;
  i = i + 1;
end

%%
t_res = {};
for i = 1:numel(r_ops)
  r = r_ops{i}(:);
  b0 = b0_ops{i}(:);
  b1 = b1_ops{i}(:);
  t = table( r, b0, b1, 'va', ["result", "b0", "b1"] );
  t.n_gates(:) = num_ops(i);
  t_res{end+1, 1} = t;
end
t = vertcat( t_res{:} );

dt = arrayfun( @(x) repmat(num_ops(x), size(d_ops{x})), 1:numel(num_ops), 'un', 0 );
dt = arrayfun( @(x) table(dt{x}, d_ops{x}, 'va', ["n_gates", "gate"]), 1:numel(num_ops), 'un', 0 );
dt = vertcat( dt{:} );
dt.v(:) = 1/num_samp;

%%

figure(1); clf;
axs = plots.summarized3( t.result, [], t(:, ["b0", "b1"]), t(:, 'n_gates'), [] ...
  , type='bar', ColorFunc=@summer );
ylim( axs, [0, 1] );

%%

figure(1); clf;
% plt = sortrows( dt, {'op_index', 'n_gates'} );
plt = dt;
axs = plots.summarized3( plt.v, [], [], plt(:, 'n_gates'), plt(:, 'gate') ...
  , type='bar', ColorFunc=@summer, SummaryFunc=@sum, NumericX=1 );
% ylim( axs, [0, 1] );

%%

%{

thinking out loud a little bit but wonder if it's more about 

%}

%%

function r = evaluate(os, b0, b1)

for i = 1:numel(os)
  r = os{i}(b0, b1);
  b1 = b0;
  b0 = r;
end

end

function ops = ops()

% Mapping of operation names → binary anonymous functions
ops = struct( ...
    'NOR',      @(x,y) double(~x & ~y),             ... % not-(x OR y)
    'Xq',       @(x,y) double(x==1 & y==0),         ... % 1 0  → 1
    'ABJ',      @(x,y) double(x==0 & y==1),         ... % 0 1  → 1
    'XOR',      @(x,y) double(x ~= y),              ... % exclusive OR
    'NAND',     @(x,y) double(~(x & y)),            ... % not-AND
    'AND',      @(x,y) double(x & y),               ... % logical AND
    'XNOR',     @(x,y) double(x == y),              ... % equality
    'if_then',  @(x,y) double(~x | (x & y)),        ... % y if x else 1
    'then_if',  @(x,y) double(~y | (x & y)),        ... % x if y else 1
    'OR',       @(x,y) double(x | y)                ... % logical OR
);

end