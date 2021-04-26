function [Wx,Wy,BB,MM] = train_ver4online(X, Y, param, L, G, H, BB, MM)


%% set the parameters
nbits = param.nbits;

theta = param.theta;
beta = param.beta;
gamma = param.gamma;

%% get the dimensions of features
n = size(X,1);

X = X';
Y = Y';
L = L';
G = G';

%% initialization
B = sign(randn(nbits, n));

%% iterative optimization
for iter = 1:param.iter
   
    % update V
    Z2 = beta*nbits* (2*([MM{1,1},zeros(nbits,1)]+B*G')*G-(MM{1,2}+B*ones(n,1))*ones(1,n)) + theta*B;
    Temp = Z2*Z2'-1/n*(Z2*ones(n,1)*(ones(1,n)*Z2'));
    [~,Lmd,QQ] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-6);
    Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
    P = (Z2'-1/n*ones(n,1)*(ones(1,n)*Z2')) *  (Q / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,(nbits-length(find(idx==1)))));
    V = sqrt(n)*[Q Q_]*[P P_]';
 
    % update B
    B = sign(nbits*H*L+beta*nbits*(2*([MM{1,3},zeros(nbits,1)]+V*G')*G-(MM{1,4}+V*ones(n,1))*ones(1,n))+theta*V);
    
end

    MM{1,1} = [MM{1,1},zeros(nbits,1)] + B*G';
    MM{1,2} = MM{1,2} + B*ones(n,1);
    MM{1,3} = [MM{1,3},zeros(nbits,1)] + V*G';
    MM{1,4} = MM{1,4} + V*ones(n,1);
    MM{1,5} = MM{1,5} + B*X';
    MM{1,6} = MM{1,6} + B*Y';
    MM{1,7} = MM{1,7} + X*X';
    MM{1,8} = MM{1,8} + Y*Y';
    MM{1,9} = MM{1,9} + B*B';

    BB{end+1,1} = B';


% update Wx
XA = MM{1,9};
XB = gamma * MM{1,7};
XC = (1+gamma) * MM{1,5};
Wx = sylvester(XA,XB,XC);
    
% update Wy
YA = MM{1,9};
YB = gamma * MM{1,8};
YC = (1+gamma) * MM{1,6};
Wy = sylvester(YA,YB,YC);
