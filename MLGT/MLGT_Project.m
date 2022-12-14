
clear

%% Starting point
a = [0.5 0.2; 0.5 0.2]' %[[0.5,0.2],[0.5,0.2]] % [[p_0,x_0], ...., [p_n,x_n]] % actions
a0 = a(:,1);
a1 = a(:,2);

og = -2;

%% Parameters
% Market parameters
market_size = 100 % k
% consumer preference
psi  = 0.105 %scaling factor

% uncertainty of the consumer willingness to pay:
mu_beta = -1.47;
sigma_beta = 1.11;
draws_size = 100;

%% Vehicle attributes
omega = 12.110; % initial manufacturing cost ($k)
gamma = 8.18; % initial operating cost (cents/mi)
tau = 241.6; % initial emissions (g/mi)
alpha = 11.554; % operating cost slope

%% COMPUTATION ==
n.F = height(a);
n.J = 1;

% Define symbolic vars
P = sym('p_J%d_F%d', [n.J n.F]);       % JxTxF tensor, vars named p_JxTxF 
Z = sym('z_J%d_F%d', [n.J n.F]);       % TechxJxTxF tensor, vars named z_JxTxF 
% Aggregate Decision Variables Array
X = [P;reshape(Z,[],n.F)];  % Decision Variables as one X

% Define dependent variables
C = (alpha * Z + omega);
oc = gamma*(1-Z); % operating cost
ghg = tau*(1-Z); % green house gases

% draw preferences
%utilityCoeffs = genUtilityCoeffs(prefParam,prefForm,n.draws);
betas = [];
for d = 1:draws_size
   betas(d,:) = normrnd(mu_beta,sigma_beta);
end

utilities = psi * (betas * gamma * (1-Z) - P); % proxy variable to model choice probability, consumer utility

demands = mean(exp(utilities)./(sum(exp(utilities),2)+og));

% leftoff: calculate market share
profits = (P-C).*demands;

% subs(profits,X,a)

pastX = a;
liveX = a;
turn = 1;
change = 1;
opts = baronset('MaxTime', 5,'MaxIter',0,'NumLoc',100);
prevTurn = 1;
xResults = a(:)';
profitResults = zeros(1,length(xResults)/2);
pastProfit = profitResults

while change > 0.1 & turn < 30
    for f = 1:n.F
        turn = turn+1;
        compIdx = 1:n.F;
        compIdx = compIdx(compIdx~=f);
        profEq = subs(profits,X(:,compIdx),pastX(:,compIdx));
        focalProfEq = profEq(f);
        focalX = num2cell(X(:,f));
        
        focalProf = matlabFunction(-focalProfEq,'Vars', {[focalX{:}]});
    
        x0 = {pastX(:,f)'};
        focalProf(x0{:});
        %min = fminunc(focalProf,x0{:}) % ,[],[])
        [min, negProfit] = baron(focalProf,[],[],[],[0 0],[100 1],[],[],[],'cc',x0{:},opts);
        profit = -negProfit
        pastX(:,f) = min';
        pastProfit(:,f) = profit

        xResults(turn,:) = pastX(:)'
        profitResults(turn,:) = pastProfit
        save times
    end
%     currentChange = results(turn,:)
    change = sum(xResults(turn,:)-xResults(prevTurn,:))
    prevTurn = turn
end
profitResults

%%
xResultsScaled = xResults
xResultsScaled(:, [1,3]) = xResults(:, [1 3])/100
plot([xResultsScaled profitResults])

subs(utilities(end,:),X(:),xResultsScaled(end,:)')
% og = 0: [-0.0448, -0.0448]
% og = 1: [-0.3605, -0.3605]

% issue with output