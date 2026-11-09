function net = buildISACAblationNetwork(cfg, variant)
%buildISACAblationNetwork Build one of the controlled ablation networks.

arguments
    cfg (1,1) struct
    variant (1,1) string {mustBeMember(variant, ...
        ["rd_only" "rd_md" "rd_md_attention"])}
end

net = dlnetwork;
net = addLayers(net, branchLayers("rd"));

if variant == "rd_only"
    head = [
        fullyConnectedLayer(cfg.EmbeddingDimension, Name="embedding_fc")
        reluLayer(Name="embedding")
        fullyConnectedLayer(numel(cfg.ClassNames), Name="classifier")
        softmaxLayer(Name="class_probs")];
    net = addLayers(net, head);
    net = connectLayers(net, "rd_feature", "embedding_fc");
    example = dlarray(zeros([cfg.ImageSize 1], "single"), "SSCB");
    net = initialize(net, example);
    return
end

net = addLayers(net, branchLayers("md"));
fusion = [
    concatenationLayer(1, 2, Name="feature_concat")
    fullyConnectedLayer(cfg.EmbeddingDimension, Name="embedding_fc")
    reluLayer(Name="embedding")
    fullyConnectedLayer(numel(cfg.ClassNames), Name="classifier")
    softmaxLayer(Name="class_probs")];
net = addLayers(net, fusion);
net = connectLayers(net, "rd_feature", "feature_concat/in1");
net = connectLayers(net, "md_feature", "feature_concat/in2");

if variant == "rd_md_attention"
    attention = [
        fullyConnectedLayer(128, Name="attention_fc")
        sigmoidLayer(Name="attention_weights")];
    net = addLayers(net, attention);
    net = addLayers(net, multiplicationLayer(2, Name="attention_multiply"));
    net = disconnectLayers(net, "feature_concat", "embedding_fc");
    net = connectLayers(net, "feature_concat", "attention_fc");
    net = connectLayers(net, "feature_concat", "attention_multiply/in1");
    net = connectLayers(net, "attention_weights", "attention_multiply/in2");
    net = connectLayers(net, "attention_multiply", "embedding_fc");
end

example = dlarray(zeros([cfg.ImageSize 1], "single"), "SSCB");
net = initialize(net, example, example);
end

function layers = branchLayers(prefix)
layers = [
    imageInputLayer([128 128 1], Normalization="none", Name=prefix+"_input")
    convolution2dLayer(5, 12, Stride=2, Padding="same", Name=prefix+"_conv1")
    batchNormalizationLayer(Name=prefix+"_bn1")
    reluLayer(Name=prefix+"_relu1")
    convolution2dLayer(3, 24, Stride=2, Padding="same", Name=prefix+"_conv2")
    batchNormalizationLayer(Name=prefix+"_bn2")
    reluLayer(Name=prefix+"_relu2")
    convolution2dLayer(3, 48, Stride=2, Padding="same", Name=prefix+"_conv3")
    batchNormalizationLayer(Name=prefix+"_bn3")
    reluLayer(Name=prefix+"_relu3")
    globalAveragePooling2dLayer(Name=prefix+"_gap")
    fullyConnectedLayer(64, Name=prefix+"_fc")
    reluLayer(Name=prefix+"_feature")];
end
