%%

%{

@TODO (6/11)

1. Confirm that, with beta = [0, 1e-2, 1e-1], the same pattern of use of
iterative processing (medium, high, low) is conserved when ponder costs are
fixed to be 1e-3, rather than rescaled with changing beta.

2. Rerun with beta in 1e-2 * [1/2, 1/3, 1/4, 2, 3, 4]. Currently, it seems
that beta > 1e-2 tends back towards beta = 0 wrt use of iterative processing.

3. Evaluate solutions of networks with simple representations. Hypothesis
is that simple solutions represent the output statistics, whereas more 
complex solutions learn the underlying algorithm.

%}

%%

addpath( '/Users/nick/source/matlab/cbrewer2/cbrewer2' );
addpath( '/Users/nick/source/matlab/28790/colorspace' );

%%

fps = shared_utils.io.findmat( ...
  '/Users/nick/source/mattarlab/explore-variational-rnn/experiment/results' );
fps = date_filter( fps, datetime('09-Jun-2025') );
ts = load_eval_results( fps );

%%

decode_fs = shared_utils.io.findmat( ...
  '/Users/nick/source/mattarlab/explore-variational-rnn/experiment/results/decoding' );
decode_fs = date_filter( decode_fs, datetime('09-Jun-2025') );
dts = load_decode_results( decode_fs );

%

% dts.pred_ys (batch x tick x sequence x prediction_type[result, operation index])

edts = evaluate_decoding_results( dts );

%%  decoding accuracy in final hidden state

figure(1); clf;
m = true( rows(edts), 1 );
m = m & edts.pred_type == "op_index";
% m = m & ismember( edts.beta, [0, 1e-1, 1e-2] );
m = m & edts.weight_normalization_type == 'none';
plt = edts(m, :);
plots.summarized3( plt.final_acc, [], plt(:, "pred_type"), plt(:, "beta"), [] ...
  , Type='bar', AddPoints=1, UseBarX=1, ColorFunc=@(n) cbrewer2('oranges', n) );
ylabel( 'acc' );

%%

% sis = [8, 10, 20, 100, 300, 400, 500];

% m = edts.pred_type == "op_index" & edts.final_acc < 0.6;
% m = edts.pred_type == "op_index" & edts.beta == 1e-2;
% m = edts.pred_type == "result" & edts.beta == 1e-2;
m = true( rows(edts), 1 );
m = m & edts.pred_type == "result";
m = m & edts.beta == 0.1;
m = m & edts.weight_normalization_type == 'none';
mf = find( m ); 
mf = mf(1);

% mf = mf(3);

edt = edts(mf, :);
true_seq_lens = cellfun( @numel, edt.true_seq{1} );
pred_seq_lens = cellfun( @numel, edt.pred_seq{1} );

sis = find( true_seq_lens > 9, 8 );
% sis = find( (true_seq_lens == pred_seq_lens) & true_seq_lens > 5, 4 );

figure(1); clf;
axs = plots.panels( numel(sis) );
for i = 1:numel(sis)
  ax = axs(i);
  true_y = edt.true_seq{1}{sis(i)};
  pred_y = edt.pred_seq{1}{sis(i)};
  h(1) = plot( ax, linspace(0, 1, numel(true_y)), true_y ...
    , 'linewidth', 2, 'displayname', 'true' ); 
  hold( ax, 'on' );
  h(2) = plot( ax, linspace(0, 1, numel(pred_y)), pred_y ...
    , 'linewidth', 1, 'displayname', 'hidden-state-pred' );
  legend( h );
  title( ax, plots.strip_underscore(table2str(edt(:, {'task_type','seed','beta'}))) );
  plots.onelegend( gcf );
end

%%  measures over learning (non-bottlenecked model)

figure(1); clf;
m = ts.step_type == 'act' & ts.bottleneck_K == 0 & ts.fixed_num_ops == -1;
plt = sortrows( ts(m, :), {'epoch', 'ponder_cost'} );
yvar = 'ticks';
% yvar = 'acc';
yvar = 'err_rate';
plots.summarized3( ...
    plt.(yvar), [], plt(:, 'beta'), plt(:, 'ponder_cost'), plt(:, 'epoch') ...
  , AddPoints=true, ColorFunc=@(n) cbrewer2('greens', n) );
ylabel( plots.strip_underscore(yvar) );

%%  ponder vs. acc

figure(1); clf;
m = ts.step_type == 'act' & ts.bottleneck_K > 0 & ts.fixed_num_ops == -1 & ...
  ts.epoch == max( ts.epoch );
% m = m & ismember( ts.beta, [0, 1e-2, 1e-1] );
m = m & ts.weight_normalization_type == 'none';
plt = sortrows( ts(m, :), {'epoch', 'ponder_cost'} );
% pv = "beta"; gv = "ponder_cost"; xv = "ticks";
pv = "ponder_cost"; gv = "beta"; xv = "ticks";
% yvar = 'err_rate';
% yvar = 'ticks';
yvar = 'acc';
% plt = summarize_across( plt(:, [pv, gv, xv, "seed", yvar]), 'seed', [xv, yvar], @mean );

cfunc = @(n) cbrewer2('oranges', n);
% cfunc = @hsv;

axs = plots.summarized3( plt.(yvar), [], plt(:, pv), plt(:, gv), plt(:, xv) ...
  , Type='scatter', ColorFunc=cfunc, MarkerSize=64 ...
  , WarnMissingX=0, summarize=0, XJitter=0 );
ylabel( plots.strip_underscore(yvar) );
% xlim( [1, 16] ); ylim( axs, [0, .3] );

%%  measures after learning

figure(2); clf;
m = ts.step_type == 'act' & ts.epoch == max( ts.epoch ) & ...
  ts.bottleneck_K > 0 & ts.ponder_cost == 1e-3;
m = m & ts.fixed_num_ops > 0;
% m = m & ts.fixed_num_ops == -1;
% m = m & ismember( ts.beta, [0, 1e-2, 2e-2, 3e-2, 1e-1] );
% m = m & ismember( ts.beta, [0, 1e-2*0.5, 1e-2*(1/3), 1e-2*(1/4)] );
m = m & ts.weight_normalization_type == 'none';
plt = sortrows( ts(m, :), {'num_fixed_ticks', 'ponder_cost', 'fixed_num_ops'} );
yvar = 'ticks';
% yvar = 'acc';
xv = 'fixed_num_ops'; gv = 'beta'; pv = 'ponder_cost';
plots.summarized3( plt.(yvar), [], plt(:, pv), plt(:, gv), plt(:, xv) ...
  , AddPoints=1, ColorFunc=@(n) cbrewer2('oranges', n) ...
  , type='bar', NumericX=0, UseBarX=1 );
ylabel( yvar );

%%  accuracy with increasing # ticks, bottlenecked model, after learning

figure(2); clf;
m = ts.step_type == 'fixed' & ts.epoch == max( ts.epoch ) & ...
  ts.bottleneck_K > 0 & ts.fixed_num_ops == -1;
m = m & ts.weight_normalization_type == 'none';
% m = m & ismember( ts.beta, [0, 1e-2, 2e-2, 3e-2, 1e-1] );
plt = sortrows( ts(m, :), {'num_fixed_ticks', 'ponder_cost'} );
plots.summarized3( ...
    plt.acc, [], plt(:, 'ponder_cost'), plt(:, 'beta'), plt(:, 'num_fixed_ticks') ...
  , AddPoints=0, ColorFunc=@(n) cbrewer2('oranges', n), Type='bar' ...
  , NumericX=0, UseBarX=1 );

%%  use of ticks or acc vs. # logic gates, bottlenecked model, after learning
% aka "acc vs difficulty"

group_beta = true;

figure(2); clf;
m = ts.step_type == 'act' & ts.epoch == max( ts.epoch ) & ts.fixed_num_ops > 0 & ...
  ts.bottleneck_K > 0;
m = m & ts.weight_normalization_type == 'none';
% m = m & ismember( ts.beta, [0, 1e-2, 1e-1] );

if ( group_beta )
  m = m & ts.ponder_cost == 1e-3;
  gv = 'beta';
  pv = 'ponder_cost';
  color_fn = @(n) cbrewer2('oranges', n);
else
  m = m & ts.beta == 0; 
  gv = 'ponder_cost';
  pv = 'beta';
  color_fn = @(n) cbrewer2('greens', n);
end

yvar = 'ticks';
% yvar = 'acc';
plt = sortrows( ts(m, :), {'num_fixed_ticks', 'ponder_cost', 'fixed_num_ops'} );
plots.summarized3( ...
    plt.(yvar), [], plt(:, pv), plt(:, gv), plt(:, 'fixed_num_ops') ...
  , AddPoints=1, ColorFunc=color_fn );
ylabel( plots.strip_underscore(yvar) );

%%  for a fixed # of ops, how do more fixed ticks change accuracy?

figure(2); clf;
m = ts.step_type == 'fixed' & ts.epoch == max( ts.epoch ) & ...
  ts.bottleneck_K > 0 & ts.fixed_num_ops > 0 & ismember(ts.beta, [0, 1e-2, 1e-1]);
m = m & ismember( ts.fixed_num_ops, [1, 3, 5, 10] );
m = m & ts.ponder_cost == 1e-3;
yvar = 'acc';
plt = sortrows( ts(m, :), {'num_fixed_ticks', 'ponder_cost', 'fixed_num_ops'} );
plots.summarized3( ...
    plt.(yvar), [], plt(:, {'beta', 'ponder_cost'}) ...
  , plt(:, 'fixed_num_ops'), plt(:, 'num_fixed_ticks') ...
  , AddPoints=1, ColorFunc=@(n) cbrewer2('blues', n) );
ylabel( yvar );

%%  for a fixed ponder cost, how does beta change the relationship between 
% ticks and accuracy?

figure(2); clf;
m = ts.step_type == 'fixed' & ts.epoch == max( ts.epoch ) & ...
  ts.bottleneck_K > 0 & ts.fixed_num_ops > 0 & ts.ponder_cost == 1e-3;
m = m & ismember( ts.fixed_num_ops, [1, 3, 5, 10] );
yvar = 'acc';
plt = sortrows( ts(m, :), {'num_fixed_ticks', 'ponder_cost', 'fixed_num_ops'} );
plots.summarized3( ...
    plt.(yvar), [], plt(:, {'ponder_cost', 'fixed_num_ops'}) ...
  , plt(:, 'beta'), plt(:, 'num_fixed_ticks') ...
  , AddPoints=1, ColorFunc=@(n) cbrewer2('oranges', n) );
%%

function t = fixup_table(t)
for j = 1:size(t, 2)
  vn = t.Properties.VariableNames{j};
  if ( isnumeric(t.(vn)) ), t.(vn) = double( t.(vn) ); end
  if ( ischar(t.(vn)) || iscellstr(t.(vn)) ), t.(vn) = string( t.(vn) ); end
end
end

function hps = fixup_hps(hps)
if ( ~isfield(hps, 'beta') ), hps.beta = 0; end
if ( ~isfield(hps, 'bottleneck_K') ), hps.bottleneck_K = 0; end
if ( ~isfield(hps, 'fixed_num_ops') ), hps.fixed_num_ops = -1; end
if ( ~isfield(hps, 'weight_normalization_type') )
  hps.weight_normalization_type = 'norm'; 
end
end

function dts = load_decode_results(decode_fs)

dts = {};
for i = 1:numel(decode_fs)
  x = load( decode_fs{i} );
  x.hps = fixup_hps( x.hps );
  hps = struct2table( x.hps );
  rest = structfun( @(x) {x}, rmfield(x, 'hps'), 'un', 0 );
  dt = [ hps, struct2table(rest) ];
  dts{end+1} = fixup_table( dt );
end
dts = vertcat( dts{:} );

end

function ts = load_eval_results(fps)

ts = {};
for i = 1:numel(fps)
  fprintf( '\n %d of %d', i, numel(fps) );
  res = load( fps{i} );
  res.hps = fixup_hps( res.hps );
  t = struct2table( res.hps );
  rest = struct2table( rmfield(res, 'hps') );
  t = [ t, rest ];
  t = fixup_table( t );
  ts{end+1, 1} = t;
end

ts = vertcat( ts{:} );

end

function edts = evaluate_decoding_results(dts)

pred_types = ["result", "op_index"];
edts = {};

for pred_type = 1:numel(pred_types)

eval_dts = dts;
eval_dts.final_acc(:) = 0;
eval_dts.pred_seq = cell( rows(dts), 1 );
eval_dts.true_seq = cell( rows(dts), 1 );
eval_dts.pred_type(:) = pred_types(pred_type);

for dti = 1:size(dts, 1)
  dt = dts(dti, :);
  
  bs = size( dt.intermediates{1}, 1 );
  pred_corr = zeros( bs, 1 );

  pred_seq = cell( bs, 1 );
  true_seq = cell( bs, 1 );

  for egi = 1:bs
    % egi = 7;
    si = 1; % sequence index
    prediction_type = pred_type; % 1: result (1/0); : operation index (1:num_ops)
    y_true_index = ternary( prediction_type == 1, 3, 1 ); % tasks.py: (op_index, b_curr, b_prev, res, B)
    
    num_ops = dt.intermediates{1}(egi, 1, 5, si);
    true_ys = dt.intermediates{1}(egi, 1:num_ops, y_true_index, si);
    pred_ys = dt.pred_ys{1}(egi, 1:dt.n{1}(egi, si)+1, si, prediction_type);

    pred_corr(egi) = pred_ys(end) == true_ys(end);
    pred_seq{egi} = pred_ys;
    true_seq{egi} = true_ys;
  end

  eval_dts.final_acc(dti) = pnz( pred_corr );
  eval_dts.pred_seq{dti} = pred_seq;
  eval_dts.true_seq{dti} = true_seq;
end

edts{end+1} = eval_dts;

end

edts = vertcat( edts{:} );

end

function [fps, f] = date_filter(fps, dt_thresh)
d = cellfun( @dir, fps );
dt = datetime( {d.date} );
f = dt > dt_thresh;
fps = fps(f);
end