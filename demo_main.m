
db_name = 'mir';
loopnbits = [8 16 32 64 128];
param.top_K = 2000;

%% load dataset
if strcmp(db_name, 'mir')
    load(['./data/',db_name,'_cnn.mat']);
    X = [I_db; I_te]; Y = [T_db; T_te]; L = [L_db; L_te];
    param.nchunks = 8;
    param.nlabels = 24;
    param.prenlabels = 17;
        
    load('./data/MIR_supplement.mat');
       
        
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    XTEChunk = cell(param.nchunks,1);
    YTEChunk = cell(param.nchunks,1);
    LTEChunk = cell(param.nchunks,1);
    for subi = 1:param.nchunks
        R = randperm(size(idx{subi,2},1));
        XChunk{subi,1} = X(idx{subi,2}(R),:);
        YChunk{subi,1} = Y(idx{subi,2}(R),:);
        LChunk{subi,1} = L(idx{subi,2}(R),Q(1:16+subi));
        XTEChunk{subi,1} = X(idx{subi,3},:);
        YTEChunk{subi,1} = Y(idx{subi,3},:);
        LTEChunk{subi,1} = L(idx{subi,3},Q(1:16+subi));
    end
        
elseif strcmp(db_name, 'nus')
    load(['./data/',db_name,'_cnn.mat']);
    X = [I_db; I_te]; Y = [T_db; T_te]; L = [L_db; L_te];
    param.nchunks = 10;
    param.nlabels = 21;
    param.prenlabels = 12;
        
    load('./data/NUS_supplement.mat');
        
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    XTEChunk = cell(param.nchunks,1);
    YTEChunk = cell(param.nchunks,1);
    LTEChunk = cell(param.nchunks,1);
    for subi = 1:param.nchunks
        R = randperm(size(idx{subi,2},1));
        XChunk{subi,1} = X(idx{subi,2}(R),:);
        YChunk{subi,1} = Y(idx{subi,2}(R),:);
        LChunk{subi,1} = L(idx{subi,2}(R),Q(1:11+subi));
        XTEChunk{subi,1} = X(idx{subi,3},:);
        YTEChunk{subi,1} = Y(idx{subi,3},:);
        LTEChunk{subi,1} = L(idx{subi,3},Q(1:11+subi));
    end
end
clear X Y L I_db I_te L_db L_te T_db T_te
    
    
%% Methods
    
    
for ii =1:length(loopnbits)
    fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
    param.nbits = loopnbits(ii);
    if strcmp(db_name, 'mir')
        fprintf('......%s start...... \n\n', 'THOR');
        param.theta=0.1;  param.beta=100;  param.gamma = 10;
        param.alpha=10;  param.iter=7;
        OURparam = param;
        evaluate(XChunk,YChunk,LChunk,XTEChunk,YTEChunk,LTEChunk,OURparam,Q,word2vector_300);
    elseif strcmp(db_name, 'nus')
        fprintf('......%s start...... \n\n', 'THOR');
        param.theta=1;  param.beta=0.001;  param.gamma = 1000;
        param.alpha=10;  param.iter=7;
        OURparam = param;
        evaluate(XChunk,YChunk,LChunk,XTEChunk,YTEChunk,LTEChunk,OURparam,Q,word2vector_300);
    end
end

   