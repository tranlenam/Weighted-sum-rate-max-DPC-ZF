function [SumRate,Precoders,gap] = szfdpcpapcbarriermethod(channel,Po)
global A effectivechannel
nTx = size(channel{1},2);
nUsers = length(channel);

effectivechannel = cell(nUsers,1); % users' effective channels
B = cell(nUsers,1);
B{1} = eye(nTx);
A = cell(nUsers,nTx); % matrix A in (33)

h_bar=[];
Nr=zeros(1,nUsers);
for iUser=1:nUsers
    h_bar = [h_bar;channel{iUser}];
    if( iUser < nUsers)
        B{iUser+1} = null(h_bar); % B{iUser} is an orthogonal basis of the composite channel from user 1 to (iUser-1)
    end
    effectivechannel{iUser} = channel{iUser}*B{iUser}; 
    for iTx=1:nTx
        A{iUser,iTx} = B{iUser}'*diag(1:nTx==iTx)*B{iUser}; % compute matrix A in (33)
    end    
end

initialOmega=cell(nUsers,1); % initial value for \Omega in (31)
for iUser=1:nUsers
    initialOmega{iUser} = eye(size(B{iUser},2)); % set to identity matrix (other values are possible)
end

MAXITERS = 100;
ALPHA = 0.01; % backtracking line search parameter
BETA = 0.5; % backtracking line search parameter
NTTOL = 1e-5;
myphi = 50;
TOL = 1e-5;

%% Initialization
Omega=cell(nUsers,1);
Omega_next=cell(nUsers,1);
DeltaOmega=cell(nUsers,1);

t=1; % the value of t0 in Algorithm 2
for iUser = 1:nUsers
    Omega{iUser} = initialOmega{iUser};
    DeltaOmega{iUser} = zeros(size(Omega{iUser}));
end
mymu = zeros(nTx,1);
mymunext = zeros(nTx,1);
iCenteringstep = 1;
iTotaliteration = 0;
sumratebest = [];

%% PHASE I
for iIteration = 1:MAXITERS
    iTotaliteration = iTotaliteration+1;
    modifiedobjval = computemodifiedobjval(t,Omega);
    sumrate(iTotaliteration) = computeobjval(Omega);
    sumratebest = [sumratebest min(sumrate)];
    sumratebarrier(iTotaliteration) = modifiedobjval;
    [DeltaOmega,Deltamu] = computeinfeasiblestartNewtonstep(t,Omega,mymu,Po); 
    residualnorm = computeresidualnorm(t,Omega,mymu,effectivechannel,A,Po);
    
    if (residualnorm < NTTOL),
        break; % if residualnorm is small, the break the for loop
    end;
    s=1;
    while(1) % Backtracking line search
        feasibility_flag = 0;
        for iUser = 1:nUsers
            feasibility_flag = feasibility_flag+(min(real(eig(Omega{iUser}+s*DeltaOmega{iUser}))) <= 0);
            Omega_next{iUser} = Omega{iUser}+s*DeltaOmega{iUser};
        end
        mymunext = mymu+s*Deltamu;
        r_norm_next = computeresidualnorm(t,Omega_next,mymunext,effectivechannel,A,Po);
        
        minimum_flag = (r_norm_next > (1-ALPHA*s)*residualnorm);
        if(feasibility_flag==0) && (minimum_flag==0)
            break
        else
            s=s*BETA;
        end
        
    end
    % Update primal and dual variabls
    mymu = mymunext;
    for iUser=1:nUsers
        Omega{iUser}=Omega_next{iUser};
    end
end
% End of the first centering step
iCenteringstep = iCenteringstep + 1;
t=t*myphi;

%% Phase II 
while(t<1e8)
    % Begin of inner loop
    for iIteration=1:MAXITERS
        iTotaliteration=iTotaliteration+1;
        [modifiedobjval,Grad_barrier] = computemodifiedobjval(t,Omega);
        [DeltaOmega,Deltamu] = computeinfeasiblestartNewtonstep(t,Omega,mymu,Po);
        sumrate(iTotaliteration) = computeobjval(Omega);
        sumratebest=[sumratebest min(sumrate)];
        sumratebarrier(iTotaliteration) = modifiedobjval;
        
        % Compute Newton decrement 
        Newtondecrement = 0;
        for iUser=1:nUsers
            Newtondecrement = Newtondecrement + ...
                real(trace(Grad_barrier{iUser}' * DeltaOmega{iUser})); 
        end
        
        % Backtracking line search
        if (abs(Newtondecrement) < NTTOL),
            % SumRate_barrier_centering(iCenteringstep) = modifiedobjval;
            break;
        end;
        
        s=1;
        while(1)
            feasibility_flag=0;
            for iUser=1:nUsers
                feasibility_flag=feasibility_flag+(min(real(eig(Omega{iUser}+s*DeltaOmega{iUser}))) <= 0);
            end
            
            for iUser=1:nUsers
                Omega_next{iUser}=Omega{iUser}+s*DeltaOmega{iUser};
            end
            mymunext=mymu+s*Deltamu;

            val_next = computemodifiedobjval(t,Omega_next);
            
            minimum_flag =(val_next > modifiedobjval + s*ALPHA*Newtondecrement);
            if(feasibility_flag==0) && (minimum_flag==0)
                break
            else
                s=s*BETA;
            end
            
        end
        % Update primal and dual variables
        mymu = mymunext;
        for iUser=1:nUsers
            Omega{iUser} = Omega_next{iUser};
        end
    end
    % End of inner loop
    iCenteringstep = iCenteringstep+1;
    t = t*myphi;
end

gap = sumratebest - sumratebest(end);
SumRate = -sumrate(end);
Precoders = Omega;

function residualnorm = computeresidualnorm(t,Omega,Deltamu,effectivechannel, A, Po)
nUsers=length(Omega);
Grad_orginal = cell(nUsers,1);
Grad_barrier = cell(nUsers,1);
rdual = cell(nUsers,1);
for iUser = 1:nUsers
    [Nr]=size(Omega{iUser},1);
    Grad_orginal{iUser} = -effectivechannel{iUser}'*effectivechannel{iUser}...
        *inv(eye(Nr)+Omega{iUser}*effectivechannel{iUser}'*effectivechannel{iUser});
    Grad_barrier{iUser} = t*Grad_orginal{iUser}-inv(Omega{iUser});
    rdual{iUser} = Grad_barrier{iUser};
end
nTx=size(Omega{1},2);
v=zeros(nTx,1);
for i = 1:nTx
    v(i) = Po(i);
    for iUser = 1:nUsers
        rdual{iUser} = rdual{iUser}+Deltamu(i)*A{iUser,i};
        v(i) = v(i)-real(trace(A{iUser,i}*Omega{iUser}));
    end
end
r = zeros(nUsers,1);
for iUser = 1:nUsers
    r(iUser) = r(iUser)+norm(rdual{iUser},'fro');
end
residualnorm = sum(r) + norm(v);



function [DeltaOmega,Deltamu] = computeinfeasiblestartNewtonstep(t,Omega,mymu,Po)
global effectivechannel A
nUsers=length(Omega);
Grad_orginal=cell(nUsers,1);
nTx=size(Omega{1},1);
myXi=cell(nUsers,nTx+1);
DeltaOmega=cell(nUsers,1);

for iUser=1:nUsers
    [Nr]=size(Omega{iUser},1);
    Grad_orginal{iUser}= -effectivechannel{iUser}'*effectivechannel{iUser}*inv(eye(Nr)+Omega{iUser}*effectivechannel{iUser}'*effectivechannel{iUser});
    
    P = t*Omega{iUser}*(-Grad_orginal{iUser});
    
    C = t*Omega{iUser}*(-Grad_orginal{iUser})*Omega{iUser}+Omega{iUser};
    for i=1:nTx
        C = C - mymu(i)*Omega{iUser}*A{iUser,i}*Omega{iUser};
    end
    myXi{iUser,1}=dlyap(P,-P'/t,C); % solve the first equation of (42)
    
    for i=1:nTx
        P=t*Omega{iUser}*(-Grad_orginal{iUser});
        C=-Omega{iUser}*A{iUser,i}*Omega{iUser};
        myXi{iUser,i+1}=dlyap(P,-P'/t,C); % solve the remaining equations of (42)
    end
end

myPsi=zeros(nTx,nTx);
mypsi=zeros(nTx,1);

for i=1:nTx
    mypsi(i)=Po(i);
    for iUser=1:nUsers
        mypsi(i) = mypsi(i)-trace(A{iUser,i} * myXi{iUser,1})-trace(A{iUser,i}*Omega{iUser});
        for j=1:nTx
            myPsi(i,j)=myPsi(i,j)+trace(A{iUser,i} * myXi{iUser,j+1});
        end
    end
end

% Newton step for dual variables
Deltamu=real(myPsi\mypsi);

% Newton step for primal variables

for iUser=1:nUsers
    DeltaOmega{iUser}=myXi{iUser,1};
    for i=1:nTx
        DeltaOmega{iUser}=DeltaOmega{iUser}+Deltamu(i)*myXi{iUser,i+1};
        DeltaOmega{iUser} = (DeltaOmega{iUser}+DeltaOmega{iUser}')/2; % eliminate numerical errors
    end
end

%%
function [val]=computeobjval(Omega)
global effectivechannel
nUsers = length(Omega);
val = 0;
for iUser = 1:nUsers
    Nr = size(effectivechannel{iUser},1);
    val = val+log(det(eye(Nr) + effectivechannel{iUser} ...
        * Omega{iUser} *  effectivechannel{iUser}'));
end
val = -real(val);

%% 
function [ modifiedobjval ,Grad_barrier] = computemodifiedobjval(t,Omega)
global effectivechannel
nUsers = length(Omega);
Grad_orginal = cell(nUsers,1);
Grad_barrier = cell(nUsers,1);

modifiedobjval = 0;
for iUser = 1:nUsers
    [Nr,Nr_bar] = size(effectivechannel{iUser});
    modifiedobjval = modifiedobjval+t*log(det(eye(Nr) + effectivechannel{iUser} * Omega{iUser} *...
        effectivechannel{iUser}'))+log(det(Omega{iUser}));
    Grad_orginal{iUser} = -effectivechannel{iUser}'*effectivechannel{iUser}*...
        inv(eye(Nr_bar)+Omega{iUser}*effectivechannel{iUser}'*effectivechannel{iUser});
    Grad_barrier{iUser} = t*Grad_orginal{iUser}-inv(Omega{iUser});
end
modifiedobjval = -real(modifiedobjval);
