function [loss, classificationLoss, contrastiveLoss, gradients, state] = ...
    isacModelGradients(net, rd, md, targets, identityMask, ...
    temperature, contrastiveWeight)
%isacModelGradients Compute classification and supervised contrastive loss.

[embedding, probabilities, state] = forward(net, rd, md, ...
    Outputs=["embedding" "class_probs"]);

probabilities = stripdims(probabilities);
targets = stripdims(targets);
batchSize = size(targets, 2);
classificationLoss = -sum(targets.*log(probabilities+1e-7), "all")/batchSize;

embedding = stripdims(embedding);
embedding = embedding./sqrt(sum(embedding.^2, 1)+1e-8);
similarity = (embedding.'*embedding)/temperature;
similarity = similarity-max(similarity, [], 2);

identityMask = stripdims(identityMask);
selfMask = 1-identityMask;
positiveMask = (targets.'*targets).*selfMask;
expSimilarity = exp(similarity).*selfMask;
logProbability = similarity-log(sum(expSimilarity, 2)+1e-8);
positiveCount = sum(positiveMask, 2);
contrastiveLoss = -sum(sum(positiveMask.*logProbability, 2)./positiveCount)/batchSize;

loss = classificationLoss+contrastiveWeight*contrastiveLoss;
gradients = dlgradient(loss, net.Learnables);
end
