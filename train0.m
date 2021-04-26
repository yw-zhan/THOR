function [Wx,Wy,BB,MM] = train0_ver4online(X, Y, param, L, G, H)


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
    Z2 = beta*nbits*(2*(B*G')*G-B*ones(n,1)*ones(1,n)) + theta*B;
    Temp = Z2*Z2'-1/n*(Z2*ones(n,1)*(ones(1,n)*Z2'));
    [~,Lmd,QQ] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-6);
    Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
    P = (Z2'-1/n*ones(n,1)*(ones(1,n)*Z2')) *  (Q / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,(nbits-length(find(idx==1)))));
    V = sqrt(n)*[Q Q_]*[P P_]';
 
    % update B
    B = sign(nbits*H*L+beta*nbits*(2*(V*G')*G-V*ones(n,1)*ones(1,n))+theta*V);
    
end

    M1 = B*G';
    M2 = B*ones(n,1);
    M3 = V*G';
    M4 = V*ones(n,1);
    M5 = B*X';
    M6 = B*Y';
    M7 = X*X';
    M8 = Y*Y';
    M9 = B*B';
    
    MM{1,1} = M1;
    MM{1,2} = M2;
    MM{1,3} = M3;
    MM{1,4} = M4;
    MM{1,5} = M5;
    MM{1,6} = M6;
    MM{1,7} = M7;
    MM{1,8} = M8;
    MM{1,9} = M9;

    BB{1,1} = B';


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

