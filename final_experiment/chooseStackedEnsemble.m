function chooseStackedEnsemble()
% Learn an ensemble of dual-branch and single-branch score vectors.
root=fileparts(mfilename('fullpath')); cfg=isacTrainingConfig();
rd=load(fullfile(root,'refined_rd_results','rd_only','best_model.mat'),'bestNet');
md=load(fullfile(root,'new_single_branch_results','md_only','best_model.mat'),'bestNet');
seeds=[20271108 20271109 20271110]; nets=cell(1,3);
for k=1:3
    z=load(fullfile(root,'refined_dual_candidates',sprintf('seed_%d',seeds(k)),'best_model.mat'),'bestNet'); nets{k}=z.bestNet;
end
cfg.DatasetRoot=fullfile(root,'dataset_rd_pooled'); d=loadISACLearningData(cfg);
ids=d.ValidationIndices; [X,~,~]=features(d,ids,cfg,rd.bestNet,md.bestNet,nets);
Y=categorical(double(d.Labels(ids)),1:3);
cfg.DatasetRoot=fullfile(root,'dataset_dev_pooled'); dev=loadISACLearningData(cfg);
ids=(1:numel(dev.Labels)).'; [Xdev,rdDev,mdDev]=features(dev,ids,cfg,rd.bestNet,md.bestNet,nets);
Ydev=categorical(double(dev.Labels),1:3); truth=double(dev.Labels);
layers=[featureInputLayer(15,Normalization='none')
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(3)
    softmaxLayer];
opts=trainingOptions('adam',ExecutionEnvironment='cpu',MaxEpochs=100, ...
    MiniBatchSize=30,InitialLearnRate=0.001,L2Regularization=0.01, ...
    ValidationData={Xdev,Ydev},ValidationFrequency=9,ValidationPatience=12, ...
    OutputNetwork='best-validation-loss',Verbose=false);
trialSeeds=[20271211 20271212 20271213]; accuracy=zeros(3,1); heads=cell(3,1);
for k=1:3
    rng(trialSeeds(k),'twister'); heads{k}=trainnet(X,Y,layers,'crossentropy',opts);
    [~,pred]=max(minibatchpredict(heads{k},Xdev),[],2);
    accuracy(k)=mean(pred==truth);
end
[~,best]=max(accuracy); fusionNet=heads{best};
out=fullfile(root,'stacked_selection'); if ~isfolder(out),mkdir(out);end
T=table(trialSeeds.',accuracy,(1:3).'==best,VariableNames={'seed','development_accuracy','selected'});
writetable(T,fullfile(out,'candidates.csv')); save(fullfile(out,'model.mat'),'fusionNet','seeds','-v7.3');
fprintf('Development RD %.4f, MD %.4f\n',mean(rdDev==truth.'),mean(mdDev==truth.')); disp(T);
end

function [X,rdPred,mdPred]=features(d,ids,cfg,rd,md,nets)
er=evaluateISACAblationNetwork(rd,d,ids,cfg,false);
tmp=d; tmp.RD=d.MD; em=evaluateISACAblationNetwork(md,tmp,ids,cfg,false);
S=[er.Scores;em.Scores]; rdPred=er.Predicted; mdPred=em.Predicted;
for k=1:numel(nets)
    e=evaluateISACNetwork(nets{k},d,ids,cfg); S=[S;e.Scores]; %#ok<AGROW>
end
X=single(S.');
end
