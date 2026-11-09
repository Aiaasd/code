function makePaperFigures_delivery()
% Figures and summary tables from the actual final 300-sample experiment.
root=fileparts(mfilename('fullpath'));
source=root;
out=fullfile(root,'paper_results','figures'); if ~isfolder(out),mkdir(out);end
data=fullfile(source,'stacked_holdout_pooled');
M=readtable(fullfile(data,'manifest.csv'),TextType='string');
P=readtable(fullfile(source,'stacked_results','predictions.csv'),TextType='string');
names=["bird" "drone" "vehicle"]; titles={'Bird','UAV','Vehicle'};
positions=zeros(3,3); maps=cell(3,2);
for i=1:3
    q=find(M.label==names(i) & M.snr_db==0,1);
    s=load(fullfile(data,strrep(M.mat_path(q),'\',filesep)),'metadata','rdMap','mdMap');
    positions(:,i)=s.metadata.Scenario.CenterPosition; maps{i,1}=s.rdMap; maps{i,2}=s.mdMap;
end

f=figure(Visible='off',Color='white',Position=[100 100 950 700]); ax=axes(f); hold(ax,'on');
tx=[0 0 0]; rx=[80 60 0]; colors=[0.15 0.4 0.7;0.8 0.3 0.1;0.18 0.58 0.3];
for i=1:3
    p=positions(:,i).';
    plot3(ax,[tx(1),p(1)],[tx(2),p(2)],[tx(3),p(3)],'-',Color=colors(i,:),LineWidth=1.2,HandleVisibility='off');
    plot3(ax,[p(1),rx(1)],[p(2),rx(2)],[p(3),rx(3)],'--',Color=colors(i,:),LineWidth=1.2,HandleVisibility='off');
    scatter3(ax,p(1),p(2),p(3),80,colors(i,:),'filled',DisplayName=titles{i});
end
scatter3(ax,tx(1),tx(2),tx(3),100,'k','p','filled',DisplayName='Transmitter');
scatter3(ax,rx(1),rx(2),rx(3),80,[.4 .4 .4],'d','filled',DisplayName='Receiver');
view(ax,-42,25); grid(ax,'on'); box(ax,'on'); xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Height (m)');
legend(ax,Location='northeast',FontSize=17); ax.FontSize=18;
exportgraphics(f,fullfile(out,'geometry.pdf'),ContentType='vector'); close(f);

f=figure(Visible='off',Color='white',Position=[100 100 1100 1050]); tl=tiledlayout(f,3,2,TileSpacing='compact',Padding='compact');
for i=1:3
    for j=1:2
        ax=nexttile(tl); imagesc(ax,maps{i,j},[0 1]); axis(ax,'xy'); colormap(ax,parula);
        if j==1, label='RD'; ylabel(ax,[titles{i} '  |  Range bin']); xlabel(ax,'Doppler bin');
        else, label='MD'; ylabel(ax,'Doppler bin'); xlabel(ax,'Time frame'); end
        title(ax,[titles{i} ' - ' label]); ax.FontSize=22;
    end
end
cb=colorbar(ax); cb.Layout.Tile='east'; cb.Label.String='Normalized log-power'; cb.FontSize=20;
exportgraphics(f,fullfile(out,'features.pdf'),ContentType='vector'); close(f);

values=zeros(4,3);
for i=1:4
    if i==4, q=P; else, q=P(P.true_label==names(i),:); end
    values(i,:)=[mean(q.rd_prediction==q.true_label),mean(q.md_prediction==q.true_label),mean(q.fusion_prediction==q.true_label)];
end
f=figure(Visible='off',Color='white',Position=[100 100 900 600]); ax=axes(f);
b=bar(ax,1:4,100*values,'grouped'); b(1).FaceColor=[.14 .37 .68]; b(2).FaceColor=[.84 .37 .15]; b(3).FaceColor=[.15 .57 .35];
for i=1:3
    offset=[1 1 4];
    text(ax,b(i).XEndPoints,b(i).YEndPoints+offset(i),compose('%.1f',b(i).YData),HorizontalAlignment='center',FontSize=15);
end
xticks(ax,1:4); xticklabels(ax,{'Bird','UAV','Vehicle','Overall'}); ylim(ax,[0 110]); yticks(ax,0:20:100);
grid(ax,'on'); ylabel(ax,'Accuracy (%)'); ax.FontSize=19;
legend(ax,{'RD only','MD only','Stacked fusion'},Location='southoutside',Orientation='horizontal',FontSize=17);
exportgraphics(f,fullfile(out,'class_comparison.pdf'),ContentType='vector'); close(f);

C=zeros(3); for i=1:3, for j=1:3, C(i,j)=sum(P.true_label==names(i) & P.fusion_prediction==names(j));end,end
snrs=unique(P.snr_db); A=zeros(numel(snrs),3); N=zeros(numel(snrs),1);
for i=1:numel(snrs)
    t=P(P.snr_db==snrs(i),:); N(i)=height(t);
    A(i,:)=[mean(t.rd_prediction==t.true_label),mean(t.md_prediction==t.true_label),mean(t.fusion_prediction==t.true_label)];
end
f=figure(Visible='off',Color='white',Position=[100 100 900 650]); ax=axes(f);
imagesc(ax,C,[0 100]); axis(ax,'image'); colormap(ax,parula); colorbar(ax); xticks(ax,1:3); yticks(ax,1:3);
xticklabels(ax,titles); yticklabels(ax,titles); xlabel(ax,'Predicted'); ylabel(ax,'True'); ax.FontSize=24;
for i=1:3, for j=1:3, if C(i,j)>50, col='k'; else, col='w'; end
        text(ax,j,i,sprintf('%d',C(i,j)),Color=col,HorizontalAlignment='center',FontSize=28,FontWeight='bold');
end,end
exportgraphics(f,fullfile(out,'confusion.pdf'),ContentType='vector'); close(f);
f=figure(Visible='off',Color='white',Position=[100 100 900 600]); ax=axes(f); hold(ax,'on');
plot(ax,snrs,100*A(:,1),'-o',LineWidth=2,MarkerSize=7,DisplayName='RD only');
plot(ax,snrs,100*A(:,2),'--s',LineWidth=2,MarkerSize=7,DisplayName='MD only');
plot(ax,snrs,100*A(:,3),'-^',LineWidth=2,MarkerSize=7,DisplayName='Stacked fusion');
grid(ax,'on'); xticks(ax,snrs); ylim(ax,[0 100]); xlabel(ax,'Target-to-noise ratio (dB)');
ylabel(ax,'Accuracy (%)'); legend(ax,Location='southeast',FontSize=17); ax.FontSize=19;
exportgraphics(f,fullfile(out,'snr.pdf'),ContentType='vector'); close(f);
T=table(snrs,N,A(:,1),A(:,2),A(:,3),VariableNames={'snr_db','count','rd_accuracy','md_accuracy','fusion_accuracy'});
writetable(T,fullfile(root,'paper_results','snr_summary.csv')); writematrix(C,fullfile(root,'paper_results','confusion.csv'));
precision=diag(C)./sum(C,1).'; recall=diag(C)./sum(C,2); f1=2*precision.*recall./(precision+recall);
K=table(names.',precision,recall,f1,VariableNames={'class','precision','recall','f1'});
writetable(K,fullfile(root,'paper_results','class_metrics.csv')); disp(K); disp(T);
end
