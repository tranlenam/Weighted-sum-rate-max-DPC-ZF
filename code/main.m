clear all
clc
rng(1)
nTx = 6; % number of users
nRxarray = [2 2 2];   %  number of receive antennas for each user, i.e.,
%                         %  nRxarray(i) is the number of rx. antennas at the ith user
                        
% nRxarray = [2 2 ];   %  number of receive antennas for each user, i.e.,
nUsers = length(nRxarray); % number of users


P = 10; % sum power, dB scale
P = 10.^(P./10);
Po = P/nTx*ones(nTx,1); % maximum power per antenna 

channel = cell(nUsers,1); % channel matrices of users

tic
for iUser=1:nUsers
    channel{iUser} = sqrt(1/2)*(randn(nRxarray(iUser),nTx) + 1i*randn(nRxarray(iUser),nTx));    
end
[sumrate,precoder,gap] = szfdpcpapcbarriermethod(channel,Po);
toc
semilogy(gap)
xlabel('Iteration')
ylabel('Residual error')
saveas(gcf,'../results/convergence.png')
%% YALMIP code for solving (31)
effectivechannel = cell(nUsers,1);
effectivechannel{1} = channel{1};
B = cell(nUsers,1);
B{1} = eye(nTx);
h_bar = [];
A = cell(nUsers,nTx); % matrix A in (33)
for iUser=1:nUsers
    h_bar = [h_bar;channel{iUser}];
    if( iUser < nUsers)
        B{iUser+1} = null(h_bar);
    end
    effectivechannel{iUser} = channel{iUser}*B{iUser};
    for iTx=1:nTx
        A{iUser,iTx}=B{iUser}'*diag(1:nTx==iTx)*B{iUser}; % compute matrix A in (33)
    end    
end


X=cell(nUsers,1);
F=[];
for iUser = 1:nUsers
    X{iUser} = sdpvar(nTx-sum(nRxarray(1:iUser-1)),nTx-sum(nRxarray(1:iUser-1)),'hermitian','complex');
    F=[F,X{iUser}>=0];
end
obj=0;
for iUser=1:nUsers
    obj=obj+logdet(eye(nRxarray(iUser)) + effectivechannel{iUser}*X{iUser}*effectivechannel{iUser}');
end

for iTx=1:nTx
    y = 0;
    for iUser=1:nUsers
        y = y + real(trace(A{iUser,iTx}*X{iUser}));
    end
    F=[F,y <= Po(iTx)];
end

ops = sdpsettings('solver','sdpt3','verbose',0);

tic

sol = optimize(F,-obj,ops);

sumrateyalmip = real(double(obj));
precodersoftware = cell(nUsers,1);
for iUser = 1:nUsers
    precodersoftware{iUser} = B{iUser}*double(X{iUser})*B{iUser}';
end
toc

norm(sumrate-sumrateyalmip)
