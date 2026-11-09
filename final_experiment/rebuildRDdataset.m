function rebuildRDdataset(inputRoot,outputRoot)
% Rebuild RD maps from stored frequency responses using local max pooling.
% MD and labels are copied unchanged; no target truth is used in RD pooling.
if ~isfolder(outputRoot), mkdir(outputRoot); end
manifest=readtable(fullfile(inputRoot,'manifest.csv'),TextType='string');
n=height(manifest);
window=hann(1024,'periodic').';
for k=1:n
    target=fullfile(outputRoot,strrep(manifest.mat_path(k),'\',filesep));
    rdfile=fullfile(outputRoot,strrep(manifest.rd_image_path(k),'\',filesep));
    mdfile=fullfile(outputRoot,strrep(manifest.md_image_path(k),'\',filesep));
    if isfile(target) && isfile(rdfile) && isfile(mdfile), continue; end
    s=load(fullfile(inputRoot,strrep(manifest.mat_path(k),'\',filesep)),'frequencyResponse','mdMap','metadata');
    h=double(s.frequencyResponse); h=h-mean(h,2);
    rt=ifft(ifftshift(h,1),256,1);
    p=abs(fftshift(fft(rt.*window,1024,2),2)).^2;
    blocks=reshape(p,2,128,8,128);
    pooled=reshape(max(max(blocks,[],1),[],3),128,128);
    db=10*log10(pooled/max(pooled(:))+eps);
    rdMap=single((max(db,-60)+60)/60);
    mdMap=s.mdMap; metadata=s.metadata;
    for q=[string(fileparts(target)),string(fileparts(rdfile)),string(fileparts(mdfile))]
        if ~isfolder(q),mkdir(q);end
    end
    save(target,'rdMap','mdMap','metadata','-v7');
    imwrite(uint8(255*rdMap),rdfile); imwrite(uint8(255*mdMap),mdfile);
    if mod(k,50)==0 || k==n, fprintf('RD rebuilt %d/%d\n',k,n); end
end
copyfile(fullfile(inputRoot,'manifest.csv'),fullfile(outputRoot,'manifest.csv'));
end
