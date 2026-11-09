function evaluateStackedHoldout()
% Evaluate fixed weights once on the independent, balanced holdout.
root=fileparts(mfilename('fullpath')); cfg=isacTrainingConfig();
cfg.DatasetRoot=fullfile(root,'stacked_holdout_pooled');
d=loadISACLearningData(cfg); ix=(1:numel(d.Labels)).';
rd=load(fullfile(root,'refined_rd_results','rd_only','best_model.mat'),'bestNet');
md=load(fullfile(root,'new_single_branch_results','md_only','best_model.mat'),'bestNet');
w=load(fullfile(root,'stacked_selection','model.mat'));
er=evaluateISACAblationNetwork(rd.bestNet,d,ix,cfg,false);
dm=d; dm.RD=d.MD; em=evaluateISACAblationNetwork(md.bestNet,dm,ix,cfg,false);
S=[er.Scores;em.Scores];
for seed=w.seeds
    a=load(fullfile(root,'refined_dual_candidates',sprintf('seed_%d',seed),'best_model.mat'),'bestNet');
    e=evaluateISACNetwork(a.bestNet,d,ix,cfg); S=[S;e.Scores]; %#ok<AGROW>
end
prob=minibatchpredict(w.fusionNet,single(S.'));
[~,pred]=max(prob,[],2); pred=pred.'; truth=er.Truth;
out=fullfile(root,'stacked_results'); if ~isfolder(out),mkdir(out);end
results=table(d.SampleID,d.SNRDB,cfg.ClassNames(truth).',cfg.ClassNames(er.Predicted).', ...
    cfg.ClassNames(em.Predicted).',cfg.ClassNames(pred).', ...
    VariableNames={'sample_id','snr_db','true_label','rd_prediction','md_prediction','fusion_prediction'});
writetable(results,fullfile(out,'predictions.csv'));
names=[cfg.ClassNames "overall"]; count=zeros(4,1); values=zeros(4,3);
for k=1:4
    if k==4, mask=true(size(truth)); else, mask=truth==k; end
    count(k)=sum(mask);
    values(k,:)=[mean(er.Predicted(mask)==truth(mask)), ...
        mean(em.Predicted(mask)==truth(mask)),mean(pred(mask)==truth(mask))];
end
summary=table(names.',count,values(:,1),values(:,2),values(:,3), ...
    VariableNames={'class','sample_count','rd_accuracy','md_accuracy','fusion_accuracy'});
writetable(summary,fullfile(out,'by_class.csv'));
f=figure(Visible='off',Color='white',Position=[100 100 1150 700]);
ax=axes(f); b=bar(ax,1:4,100*values,'grouped');
b(1).FaceColor=[.14 .37 .68]; b(2).FaceColor=[.84 .37 .15]; b(3).FaceColor=[.15 .57 .35];
for k=1:3
    text(ax,b(k).XEndPoints,b(k).YEndPoints+1,compose('%.1f',b(k).YData), ...
        HorizontalAlignment='center',FontSize=10);
end
xticks(ax,1:4); xticklabels(ax,{'Bird','UAV','Vehicle','Overall'});
ylim(ax,[0 108]); yticks(ax,0:20:100); grid(ax,'on'); ylabel(ax,'Classification accuracy (%)');
legend(ax,{'RD only','MD only','RD+MD stacked fusion'},Location='southoutside',Orientation='horizontal');
title(ax,'Independent simulated test (100 per class)');
exportgraphics(f,fullfile(out,'comparison_by_class.pdf'),ContentType='vector');
exportgraphics(f,fullfile(out,'comparison_by_class.png'),Resolution=300);
disp(summary);
end
