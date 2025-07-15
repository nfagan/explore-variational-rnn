base_p = "/Volumes/external4/data/mattarlab/explore-variational-rnn/";
res_mats = shared_utils.io.findmat( fullfile(base_p, 'results') );
d = cellfun( @dir, res_mats );
dts = datetime( {d.date} );

fst = @(x) x(1);
reject = false( size(res_mats) );
for i = 1:numel(res_mats)
  s = strsplit( res_mats{i}, 'beta_' );
  beta_s = fst( string(strsplit(s{2}, '-')) );
  beta = double( beta_s );
  reject(i) = beta < 1e-4 & strlength( beta_s ) > 4;
  if ( 1 )
    reject(i) = reject(i) | dts(i) < datetime('19-May-2025');
  elseif ( 0 )
    % old
    reject(i) = reject(i) | dts(i) > datetime('04-May-2025');
  else
    % new
    reject(i) = reject(i) | ~contains( string(res_mats{i}), 'seed' ); % mult seeds
    reject(i) = reject(i) | ~contains( string(res_mats{i}), '-99.' ); % last epoch
    % if ( contains(string(res_mats{i}), 'hd_64') )
    %   reject(i) = false;
    % end
  end
end

[ts, uv, vv] = load_in( res_mats(~reject) );
ts.acc = 100 - ts.err;
% store_ts = ts; store_uv = uv; store_vv = vv;
% ts = store_ts; uv = store_uv; vv = store_vv;

%%  per-example I(Y; Z)

% max p
[p1, am] = max( ts.py, [], 3 );
is_corr = (am-1) == ts.y;
cl = -log( p1 );
err = 1 - sum( is_corr, 2 ) / size( ts.y, 2 );
ent = discrete_entropy( ts(1, :).y, 10 );
mi_yzs = ent - cl;
mi_yz = mean( mi_yzs, 2 );
ts.alt_mi_yz = mi_yz;
vv = union( vv, "alt_mi_yz" );

%%  2nd highest p

pys = reshape( ts.py, [], size(ts.py, 3) );
[~, ord] = sort( pys, 2, 'descend' );
% 2nd highest p
p2 = reshape( pys(ord(:, 2)), size(am) );
ord = reshape( ord, size(ts.py) );
assert( isequal(ord(:, :, 1), am) );
pdiff = p1 - p2;

%%  entropy

ts.ent = -ts.py .* log( ts.py );
ts.ent(ts.py == 0) = 0;
ts.ent = mean( sum(ts.ent, 3), 2 );
vv = union( vv, "ent" );

%%

ps = ts.py;
ty = reshape( ts.y, [], 1 );
r = reshape( ps, [], size(ps, 3) );
errs = zeros( size(r, 1), 1 );
for i = 1:size(r, 1)
  d = randsample( 0:9, 1e2, true, r(i, :) );
  errs(i) = 1 - pnz( d == ty(i) );
end
ts.alt_mc_err = mean( reshape(errs, size(ts.y)), 2 );
vv = union( vv, "alt_mc_err" );

%%  best performance for a given # ticks experienced during training

% ------------------------------------------------------------------------
m = true( rows(ts), 1 ); 
m = m & ismember( ts.epoch, 99 );
m = m & ~ismember( ts.encoder_hidden_dim, 1024 );
m = m & ismember( ts.beta, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0] );
m = m & ~ismember( ts.max_num_ticks, 2 );
m = m & ts.rand_ticks;
% ------------------------------------------------------------------------

pvar = [ "encoder_hidden_dim", "rand_ticks", "full_cov" ];
gvar = "max_num_ticks";
xvar = "beta";
yvar = "acc";

tot_v = [pvar, gvar, xvar, "seed"];
t = summarize_within( ts(m, :), tot_v, yvar, @max );
% t = ts(m, :);

figure(1); clf;
[I, pl, gl, xl] = rowsets3( t, pvar, gvar, xvar );
[~, ord] = sortrows( [gl, pl, xl] ); 
[I, pl, gl, xl] = rowref_many( ord, I, pl, gl, xl );
axs = plots.summarized3( t.(yvar), I, pl, gl, xl, ColorFunc=@summer ...
  , type='bar', NumericX=false, AddPoints=1, UseBarX=true, MarkerSize=16 );
% ylabel( axs(1), yvar ); ylim( axs, [40, 98] );
set( axs, 'xticklabelrotation', 20 );

%%  performance along the diagonal of # ticks (train on M, test on M)

% ------------------------------------------------------------------------
m = true( rows(ts), 1 ); 
m = m & ismember( ts.epoch, 99 );
% m = m & ts.encoder_hidden_dim == 1024;
m = m & ts.encoder_hidden_dim ~= 1024;
% m = m & ts.encoder_hidden_dim == 32;
m = m & ismember( ts.beta, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0] );
m = m & ts.rand_ticks;

% -- select where ticks == max_num_ticks
[I, mn] = rowgroups( ts(:, 'max_num_ticks') );
for i = 1:numel(I)
  I{i} = intersect( I{i}, find(ts.ticks == mn.max_num_ticks(i)) );
end
mt = trueat( size(m), vertcat(I{:}) );
m = m & mt;
% --
% ------------------------------------------------------------------------

assert( isequal(unique(ts.epoch(m)), max(ts.epoch)) );

pvar = {'epoch', 'beta', 'rand_ticks'};
gvar = "encoder_hidden_dim";

pvar = {'encoder_hidden_dim', 'rand_ticks'};
gvar = "beta";

xvar = 'max_num_ticks';
yvar = 'acc';

figure(1); clf;
[I, pl, gl, xl] = rowsets3( ts, pvar, gvar, xvar, mask=m );
axs = plots.summarized3( ts.(yvar), I, pl, gl, xl ...
  , ColorFunc=@summer, type='shaded-line', NumericX=true );
% set( axs, 'xticklabelrot', 10 );
% ylim( axs, [86, 98] );

%%  performance over # ticks (train on M, test on N)

% add 'split' as a gv if it is changed
assert( unique(ts.split) == 'test' );

% ------------------------------------------------------------------------
m = true( rows(ts), 1 );
m = m & ts.beta < 0.1;
% m = m & ~isnan( ts.mc_err );
% m = m & ts.max_num_ticks ~= 2;
m = m & ts.split == 'test';
% m = m & ismember( ts.epoch, [0, 50, 99] );
m = m & ismember( ts.epoch, [99] );
% m = m & ~ismember( ts.beta, [0] );
% m = m & ts.max_num_ticks == 6;
m = m & ismember( ts.max_num_ticks, [2, 4, 6, 8] );
% m = m & ismember( ts.max_num_ticks, [2, 6, 10, 14] );
m = m & ~ismember( ts.encoder_hidden_dim, [1024] );
m = m & ts.rand_ticks;
% m = m & ismember( ts.beta, [1e-3, 1e-4, 1e-2] );
% m = m & ismember( ts.beta, [0] );

exper_betas = [0.25, 0.5, 1.5, 2, 4].*1e-3;
m = m & ~ismember( ts.beta, exper_betas );
% m = m & ismember( ts.beta, [1e-3, exper_betas] ) & ts.max_num_ticks == 6;

pv =  {'max_num_ticks', 'epoch', 'beta', 'rand_ticks'};
gvar = 'encoder_hidden_dim';

pv =  {'max_num_ticks', 'epoch', 'encoder_hidden_dim', 'rand_ticks'};
gvar = 'beta';

xvar = 'ticks';
yvar = 'acc';
% yvar = 'aniso';
% yvar = 'mi_xz';
% yvar = 'err';
% yvar = 'mc_err';
% yvar = 'alt_mi_yz';
% yvar = 'alt_mc_err';
% yvar = 'ent';
% yvar = 'aniso';

figure(2); clf;

[I, pl, gl, xl] = rowsets3( ts, pv, gvar, xvar, mask=m );
[axs, info] = plots.summarized3( ts.(yvar), I, pl, gl, xl, colorfunc=@summer );

for i = 1:numel(axs)
h = shared_utils.plot.add_vertical_lines( axs(i), info(i).pl.max_num_ticks );
set( h, 'linewidth', 2 );
end

%%  replicate fig 1 of VIB paper

% ------------------------------------------------------------------------
m = true( rows(ts), 1 ); 
m = m & ismember( ts.epoch, max(ts.epoch) );
m = m & ts.encoder_hidden_dim == 1024;
m = m & ~ts.full_cov;
m = m & ts.ticks == 1 & ts.max_num_ticks == 1;
% ------------------------------------------------------------------------

assert( isequal(unique(ts.epoch(m)), max(ts.epoch)) );

pvar = {'rand_ticks', 'full_cov'};
gvar = 'encoder_hidden_dim';
xvar = 'beta';
yvar = 'mc_err';

figure(1); clf;
[I, pl, gl, xl] = rowsets3( ts, pvar, gvar, xvar, mask=m );
axs = plots.summarized3( ts.(yvar), I, pl, gl, xl ...
  , colorfunc=@summer, NumericX=0, AddPoints=1, MarkerSize=16, type='shaded-line' );
ylabel( axs, stripu(yvar) );
ylim( axs, [1, 3] );

%%  grid of beta vs. ticks

do_append = true;
if ( 1 )
  ims = {};
end

% ------------------------------------------------------------------------
m = true( rows(ts), 1 );
% m = m & ts.beta < 0.1;
% m = m & ts.beta > 0;
% m = m & ismember( ts.max_num_ticks, [6] );
% m = m & ts.ticks > 1;
% m = m & ismember( ts.beta, 10.^(-6:-1) );
exper_betas = [0.25, 0.5, 1.5, 2, 4].*1e-3;
m = m & ~ismember( ts.beta, exper_betas );
m = m & ts.epoch == max( ts.epoch );
m = m & ts.encoder_hidden_dim == 1024;
% m = m & ts.beta == 1e-3;
m = m & ismember( ts.beta, [1e-5, 1e-4, 1e-3, 1e-2] );
% m = m & ts.beta > 0;
% m = m & ts.beta > 1e-4 & ts.beta < 1e-2;
% ------------------------------------------------------------------------

% gv = {'epoch', 'task_type', 'split', 'max_num_ticks'};
% xvar = 'beta';
% yvar = 'ticks';

gv = {'beta', 'max_num_ticks', 'epoch'};
xvar = 'ticks';
yvar = 'encoder_hidden_dim';

% gv = {'encoder_hidden_dim', 'max_num_ticks', 'epoch'};
gv = {'encoder_hidden_dim', 'beta', 'epoch'};
xvar = 'max_num_ticks';
% yvar = 'beta';
yvar = 'ticks';

% zvar = 'mi_yz';
zvar = 'acc';
% zvar = 'aniso';

highlight_op = @max;

% tform_t = @(t) -log(t/100);
tform_t = @(t) t;
if ( string(zvar) == "err") , tform_t = @(t) -log(t/100); end
if ( ismember(string(zvar), ["mi_xz"]) ) , tform_t = @log; end

x = unique( ts.(xvar)(m) );
y = unique( ts.(yvar)(m) );

[I, grid] = rowgroups( ts(:, gv), m );

grid.(zvar) = zeros( rows(grid), numel(y), numel(x) );
for i = 1:numel(I)
  t = make_grid( ts(I{i}, :), xvar, yvar, zvar, x, y );
  t = tform_t( t );
  grid.(zvar)(i, :, :) = t;
end

% ------------------------------------------------------------------------
m = true( rows(grid), 1 );
m = m & ismember( grid.epoch, [0, 50, max(grid.epoch)] );
% ------------------------------------------------------------------------

[I, C] = rowsetsn( grid, {gv}, mask=m, un=true );
% [I, C] = rowgroups( grid(:, gv), m );
z = grid.(zvar);

figure(3); clf;
axs = plots.panels( numel(I) );
for i = 1:numel(I)
  ax = axs(i);
  zv = squeeze( z(I{i}, :, :) );
  assert( ismatrix(zv), 'some combinations unaccounted for' );
  do_imagesc( axs(i), zv, x, y, stripu(xvar), stripu(yvar), stripu(zvar) );
  [mx, mi] = highlight_op( zv, [], 1 );
  [mxx, mj] = highlight_op( mx ); assert( mxx == highlight_op(zv(:)) );
  hold( axs(i), 'on' ); 
  if ( 1 ), scatter( axs(i), x(:), mi, 'go', 'filled' ); end
  if ( 1 ), scatter( axs(i), mj, mi(mj), 'ro', 'filled' ); end
  tl = stripu( table2str(C(i, :)) );
  tl = compose( "%s (best = %0.3f)", tl, best_v );
  title( axs(i), tl );
  set( axs(i), 'YDir', 'normal' );
  if ( 1 )
    dx = unique( diff(double(x)) );
    dy = unique( diff(double(y)) );
    ps = { 'linewidth', 2 };
    hx = dx * 0.5; hy = dy * 0.5;
    for xi = 1:numel(x)
      for yi = 1:numel(y)
        if ( xi ~= yi ), continue; end
        x0 = double(x(xi)); x1 = x0 + dx;
        y0 = double(y(yi)); y1 = y0 + dy;
        plot( ax, [x0-hx, x0-hx], [y0-hy, y1-hy], 'k', ps{:} );
        plot( ax, [x1-hx, x1-hx], [y0-hy, y1-hy], 'k', ps{:} );
        plot( ax, [x0-hx, x1-hx], [y0-hy, y0-hy], 'k', ps{:} );
        plot( ax, [x0-hx, x1-hx], [y1-hy, y1-hy], 'k', ps{:} );
      end
    end
  end
end

% shared_utils.plot.match_clims( axs );
if ( strcmp(zvar, 'acc') )
  shared_utils.plot.set_clims( axs, [85, 98] );
elseif ( strcmp(zvar, 'mi_yz') )
  shared_utils.plot.set_clims( axs, [0.6, 0.82] );
  % shared_utils.plot.set_clims( axs, [-5, 25] );
end

annotate_colorbar( axs, stripu(zvar) );

if ( do_append )
  im = getframe ( gcf );
  [A,map] = rgb2ind( im.cdata, 256 );
  ims{end+1} = struct( 'im', A, 'map', map );
end

%%

ims_dst = {};
nl = 30;
nh = 30;
rf = 1;

for i = 1:numel(ims)
  r0 = imresize( ind2rgb(ims{i}.im, ims{i}.map), rf );
  if ( i + 1 <= numel(ims) )
    r1 = imresize( ind2rgb(ims{i+1}.im, ims{i+1}.map), rf );
  end

  t = ims{i};
  im = ind2rgb( t.im, t.map );
  t.im = imresize( im, rf );
  [t.im, t.map] = rgb2ind( t.im, 256 );

  for j = 1:nh
    ims_dst{end+1} = t;
  end
  if ( i + 1 <= numel(ims) )
    for j = 0:nl-1
      t = j / nl;
      f = r0 * (1 - t) + r1 * t;
      [r2, map] = rgb2ind( f, 256 );
      ims_dst{end+1} = struct( 'im', r2, 'map', map );
    end
  end
end
%

file_p = fullfile( base_p, "plots/summary" );
fname = compose( "heat_x-%s_y-%s_z-%s.gif", xvar, yvar, zvar );
file_p = fullfile( file_p, fname );

ims_save = ims_dst;

dt = 1/30;
for idx = 1:numel(ims_save)
  im = ims_save{idx};
  A = im.im;
  map = im.map;
  if idx == 1
    imwrite( A, map, file_p, "gif", LoopCount=Inf, DelayTime=dt )
  else
    imwrite( A, map, file_p, "gif", WriteMode="append", DelayTime=dt )
  end
end

%%  performance over learning

% ------------------------------------------------------------------------
m = true( rows(ts), 1 );
m = m & ts.beta < 0.1;
m = m & ts.split == 'test';
m = m & ~ismember( ts.beta, [0] );
m = m & ismember( ts.max_num_ticks, [6] );
% m = m & ismember( ts.ticks, max(ts.ticks) );

exper_betas = [0.25, 0.5, 1.5, 2, 4].*1e-3;
m = m & ~ismember( ts.beta, exper_betas );
% ------------------------------------------------------------------------

figure(1); clf;

pvar = {'split', 'max_num_ticks', 'beta', 'encoder_hidden_dim'};

xvar = 'epoch';
% yvar = 'mi_yz';
% yvar = 'mi_xz';
yvar = 'err';
% yvar = 'mc_err';
% yvar = 'alt_mi_yz';
% yvar = 'alt_mc_err';
% yvar = 'ent';

[I, tl] = rowgroups( ts(:, pvar), m );
tlabs = t2s( tl );
axs = plots.panels( numel(I) );

for axi = 1:numel(axs)
ax = axs(axi); axes( ax );
[lI, ls] = rowgroups( ts(:, gvar), I{axi} );
xs = cate1( cellfun(@(x) ts.(xvar)(x)', lI, 'un', 0) );
for j = 1:size(xs, 1)
  assert( numel(unique(xs(j, :))) == numel(xs(j, :)) ); 
end
ys = cate1( cellfun(@(x) ts.(yvar)(x)', lI, 'un', 0) );
h = plot_lines( xs', ys', t2s(ls), 'epoch', yvar, tlabs(axi) ...
  , 'colorfunc', @jet );
end

match_ylims_by( axs, tl, {} )
plots.onelegend( gcf );

%%

% why are both I(y;z) and error highest for smallest beta? Seemingly,
% when I(y;z) is highest, error should be lowest.

% ------------------------------------------------------------------------
m = true( rows(ts), 1 );
m = m & ts.epoch == max( ts.epoch ) & ts.max_num_ticks == 6;
m = m & ismember( ts.beta, 10.^(-4:-2) );
% ------------------------------------------------------------------------

b0 = 1e-4;
b1 = 1e-3;
beta_lo = m & ts.beta == b0;
beta_hi = m & ts.beta == b1;

mi_yz_lo = mi_yzs(beta_lo, :);
mi_yz_hi = mi_yzs(beta_hi, :);

pdiff_lo = pdiff(beta_lo, :);
pdiff_hi = pdiff(beta_hi, :);

same_pred = am(beta_lo, :) == am(beta_hi, :);
diff_pred = ~same_pred;
beta_lo_corr = is_corr(beta_lo, :);
beta_hi_corr = is_corr(beta_hi, :);

case0 = beta_lo_corr & beta_hi_corr;
case1 = beta_lo_corr & ~beta_hi_corr;
case2 = ~beta_lo_corr & beta_hi_corr;
case3 = ~beta_lo_corr & ~beta_hi_corr;

cases = {case0, case1, case2, case3};
bloc = [true, true, false, false];
bhic = [true, false, true, false];

bt = table();
for i = 1:numel(cases)
  mu_lo = mean( mi_yz_lo(cases{i}) );
  mu_hi = mean( mi_yz_hi(cases{i}) );
  vn = arrayfun( @compose, ["mi_yz_beta_%0.4f", "mi_yz_beta_%0.4f"], [b0, b1] );
  t = table( mu_lo, mu_hi, 'va', vn );
  t.(compose("corr_beta_%0.4f", b0)) = bloc(i);
  t.(compose("corr_beta_%0.4f", b1)) = bhic(i);
  t.(compose("pdiff_beta_%0.4f", b0)) = mean( pdiff_lo(cases{i}) );
  t.(compose("pdiff_beta_%0.4f", b1)) = mean( pdiff_hi(cases{i}) );
  t.p = pnz( cases{i} );
  bt = [ bt; t ];
end

bt

%%

%{

Marginal improvements in predictive information are evident with additional 
processing time. 

Complexity regularization acts as an overall scale on the maximum 
performance.

%}

%%

function vs = vnames(t)
vs = t.Properties.VariableNames;
end

function [t, I] = rowify_vars(T, rest, varargin)
rest = string( rest );
[I, t] = rowgroups( T(:, setdiff(vnames(T), rest)), varargin{:} );

for i = 1:numel(rest)
  t.(rest(i)) = cate1( cellfun(@(x) reshape_var(T.(rest(i)), x), I, 'un', 0) );
end

function dv = reshape_var(v, i)
  clns = colons( ndims(v) - 1 );
  subv = v(i, clns{:});
  dv = reshape( subv, [1, size(subv)] );
end
end

function [ts, uv, vv] = load_in(res_mats)

ts = {};
for i = 1:numel(res_mats)
  fprintf( '\n %d of %d', i, numel(res_mats) );
  res = load( res_mats{i} );
  % if ( ~isfield(res.hp, 'max_num_ticks') ), res.hp.max_num_ticks = 4; end
  % if ( ~isfield(res.hp, 'split') ), res.hp.split = "test"; end
  % if ( ~isfield(res.hp, 'task_type') ), res.hp.task_type = "mnist"; end
  ticks = res.ticks(:);
  mi_xz = res.mis(1, :)';
  mi_yz = res.mis(2, :)';
  area = res.areas(:);
  aniso = res.anisos(:);
  err = res.errs(:) * 100;
  cov = res.covs';
  cxx = cov(:, 1);
  cyy = cov(:, 2);
  cxy = cov(:, 3);

  if ( ~isfield(res, 'mc_errs') )
    mc_err = nan( size(err) );
  else
    mc_err = res.mc_errs(:) * 100;
  end
  hps = struct2table( repmat(res.hp, numel(ticks), 1) );
  if ( ~isfield(res.hp, 'encoder_hidden_dim') )
    hps.encoder_hidden_dim(:) = 1024;
  end
  if ( ~isfield(res.hp, 'rand_ticks') )
    hps.rand_ticks(:) = true;
  end
  if ( ~isfield(res.hp, 'seed') )
    hps.seed(:) = -1;
  end
  py = res.py;
  y = repmat( res.y(:)', size(py, 1), 1 );
  d = dir(res_mats{i});
  date = repmat( datetime(d.date), size(area) );

  t = table( ticks, mi_xz, mi_yz, area, aniso, err, mc_err, py ...
    , y, cxx, cyy, cxy, date );

  if ( 1 )
    t.py = [];
    t.y = [];
  end

  % uniform vars
  uv = string( hps.Properties.VariableNames(:)' );
  % varying vars
  vv = string( t.Properties.VariableNames(:)' );

  t = [ t, hps ];
  ts{end+1, 1} = t;
end

ts = vertcat( ts{:} );

for i = 1:size(ts, 2)
  if ( iscellstr(ts{:, i}) )
    ts.(ts.Properties.VariableNames{i}) = string( ts{:, i} );
  end
end

end

function annotate_colorbar(axs, zvar)
for i = 1:numel(axs)
  c = colorbar( axs(i) );
  lims = get( c, 'ticklabels' );
  lims{end} = sprintf( '%s=%s', zvar, lims{end} );
  set( c, 'ticklabels', lims );
end
end

function do_imagesc(ax, zv, x, y, xvar, yvar, zvar)

imagesc( ax, zv );
set( ax, 'xtick', 1:numel(x), 'ytick', 1:numel(y) );
set( ax, 'XTickLabel', string(x), 'YTickLabel', string(y) );
xlabel( ax, xvar ); ylabel( ax, yvar );
c = colorbar( ax );

end

function t = make_grid(ts, xvar, yvar, zvar, x, y)

t = nan( numel(y), numel(x) );
for j = 1:numel(x)
  for k = 1:numel(y)
    ind = ts.(xvar) == x(j) & ts.(yvar) == y(k);
    if ( nnz(ind) == 0 ), continue; end
    assert( nnz(ind) == 1, 'some combinations unaccounted for.' );
    t(k, j) = ts.(zvar)(ind);
  end
end

end

function r = discrete_entropy(ys, nc)
fy = zeros(nc, 1);
for i = 0:nc-1, fy(i+1) = sum(ys == i); end
fy = fy / sum( fy );
hy = -(fy .* log(fy));
hy(fy == 0) = 0;
r = sum( hy );
end

function h = plot_lines(xs, ys, ls, xlab, ylab, tlab, options)
arguments
  xs, ys, ls, xlab, ylab, tlab;
  options.ColorFunc = @hsv;
  options.ax = gca;
end
ax = options.ax;
h = plot( ax, xs, ys, 'linewidth', 2 );
cs = options.ColorFunc( numel(h) );
for i = 1:numel(h), set(h(i), 'color', cs(i, :), 'displayname', ls(i) ); end
xlabel( ax, xlab ); ylabel( ax, ylab ); legend( h );
title( ax, tlab );
end

function match_ylims_by(axs, tl, mv)
assert( numel(axs) == rows(tl), 'Mismatch between axes and rows of table' );
matchi = rowgroups( tl(:, mv) );
for i = 1:numel(matchi)
  shared_utils.plot.match_ylims( axs(matchi{i}) );
end
end

function s = t2s(t)
s = plots.strip_underscore( table2str(t) );
end

function s = stripu(s)
s = plots.strip_underscore( s );
end

function y = trueat(s, i)
y = false( s );
y(i) = true;
end