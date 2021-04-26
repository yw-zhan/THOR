function evaluate(XChunk,YChunk,LChunk,XTEChunk,YTEChunk,LTEChunk,param,Lindex,wv)
    
    nbits = param.nbits;
    nlabels = param.nlabels;
    prenlabels = param.prenlabels;
    HH = zeros(nbits,nlabels);
    K = hadamard(128);
    if nbits==128
        WW = eye(128);
    else
        WW = randn(nbits,128);
    end
    
    for chunki = 1:param.nchunks
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        GTrain_new = NormalizeFea(LTrain_new,1);
        
        XTest = XTEChunk{chunki,:};
        YTest = YTEChunk{chunki,:};
        LTest = LTEChunk{chunki,:};
        c = size(LTest,2);
        
        if (size(LTest,2)-prenlabels) == 0
            LTrain = LChunk{chunki,:};
        else
            LTrain = [[LTrain,zeros(size(LTrain,1),(c-prenlabels))];LChunk{chunki,:}];
        end
        
        alpha = param.alpha;
       
        
        % Hash code learning  

        if chunki == 1
            Li = Lindex(1:c);
            w = wv(Li,:);
            d = 1-pdist(w,'cosine');
            A2 = squareform(d) + eye(size(w,1));
            A = A2;
            KK = WW * K(:,1:c);
            Z = nbits*KK*A + alpha*KK;

            Temp = Z*Z'-1/c*(Z*ones(c,1)*(ones(1,c)*Z'));
            [~,Lmd,QQ] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            P = (Z'-1/c*ones(c,1)*(ones(1,c)*Z')) *  (Q / (sqrt(Lmd(idx,idx))));
            if c>nbits-length(find(idx==1))
                P_ = orth(randn(c,(nbits-length(find(idx==1)))));
            else
                P_ = orth(randn((nbits-length(find(idx==1))),c))';
            end
            H = sqrt(c)*[Q Q_]*[P P_]';
            HH(:,1:c) = H;
            [Wx,Wy,BB,MM] = train0(XTrain_new,YTrain_new,param,LTrain_new,GTrain_new,H);
        else
            old_Li = Lindex(1:prenlabels);
            new_Li = Lindex(prenlabels+1:c);
            old_w = wv(old_Li,:);
            new_w = wv(new_Li,:);
            Ann = 1- pdist2(new_w,new_w,'cosine');
            Ano = 1- pdist2(old_w,new_w,'cosine');

            KK_new = WW * K(:,prenlabels+1:c);
            KK_old = WW * K(:,1:prenlabels);
            Z = nbits*(KK_new*Ann+KK_old*Ano) + alpha*KK_new;
            
            c_new = c-prenlabels;

            Temp = Z*Z'-1/c_new*(Z*ones(c_new,1)*(ones(1,c_new)*Z'));
           [~,Lmd,QQ] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            P = (Z'-1/c_new*ones(c_new,1)*(ones(1,c_new)*Z')) *  (Q / (sqrt(Lmd(idx,idx))));
            if c_new>nbits-length(find(idx==1))
                P_ = orth(randn(c_new,(nbits-length(find(idx==1)))));
            else
                P_ = orth(randn((nbits-length(find(idx==1))),c_new))';
            end
            H = sqrt(c_new)*[Q Q_]*[P P_]';
            HH(:,prenlabels+1:c) = H;
            [Wx,Wy,BB,MM] = train(XTrain_new,YTrain_new,param,LTrain_new,GTrain_new,HH(:,1:c),BB,MM);
        end
        
        B = cell2mat(BB(1:end,:));
        
        prenlabels = c;
        
    end
    %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx' >= 0);
    ByTrain = compactbit(B >= 0);
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    Image_VS_Text_MAP  = mAP(orderH', LTrain, LTest);

    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Wy' >= 0);
    BxTrain = compactbit(B >= 0);
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    fprintf('Image-to-Text,MAP:%f    Text-to-Image,MAP:%f \n',Image_VS_Text_MAP,Text_VS_Image_MAP);
end
