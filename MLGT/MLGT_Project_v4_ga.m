
% clear
% rng('default')

%% Starting point

a = [50 0]' % 1 veh
a = [50 0; 50 0]' % 2 veh
a = [50 0; 50 0; 50 0]' % 3 veh
%a = [50 0.2; 50 0.2; 50 0.2; 50 0.2]' % 3 veh
ghgLim = 50;
isBev = 1;
n.F = 1;

%%
og = exp(-5) %-> -4.2480, -4.2781, -4.2780, -4.2478 % 0 ->[-4.4681, -4.4171, -4.4679, -4.4170]

n.J = width(a)/n.F;

lb = zeros(1,numel(a)/n.F);
ub = ones(1,numel(a)/n.F)*.75;
ub(1:2:end) = 75;
cont = repelem('c',length(lb));

% automate starting point, BARON bounds, etc to auto match
%% Parameters
% Market parameters
market_size = 100 % k
% consumer preference
psi  = 0.105; %scaling factor

% uncertainty of the consumer willingness to pay:
mu_beta = -1.47;
sigma_beta = 1.11;
draws_size = 10;

%% Vehicle attributes
omega = 12.110; % initial manufacturing cost ($k)
gamma = 8.18; % initial operating cost (cents/mi)
tau = 241.6; % initial emissions (g/mi)
alpha = 11.554; % operating cost slope

%% COMPUTATION ==

% Define symbolic vars
P = sym('p_J%d_F%d', [n.J n.F]);       % JxTxF tensor, vars named p_JxTxF 
Z = sym('z_J%d_F%d', [n.J n.F]);       % TechxJxTxF tensor, vars named z_JxTxF 
P = reshape(P,1,[]);
Z = reshape(Z,1,[]);
% Aggregate Decision Variables Array
X = [P;Z];  % Decision Variables as one X

% Define dependent variables
C = (alpha * Z + omega);
oc = gamma*(1-Z); % operating cost
vehGhg = tau*(1-Z); % green house gases

%% BEV Case

if isBev == 1
    ub(2) = 0;
    C(1) = 24.793;
    oc(1) = 1.71;
    vehGhg(1) = 0;
end

%% Draw preferences
%utilityCoeffs = genUtilityCoeffs(prefParam,prefForm,n.draws);
betas = [];
for d = 1:draws_size
   betas(d,:) = normrnd(mu_beta,sigma_beta);
end

%% Equations
utilities = psi * (betas * gamma * (1-Z) - P); % proxy variable to model choice probability, consumer utility
demands = mean(exp(utilities)./(sum(exp(utilities),2)+og));
vehProfits = (P-C).*demands;
profits = sum(reshape(vehProfits,n.J,n.F));
swaGhg = sum(reshape(vehGhg.*demands,n.J,n.F))/sum(reshape(demands,n.J,n.F),1);

%% SIO
pastX = a;
liveX = a;
turn = 1;
change = 1;
opts = baronset('MaxTime', 5,'MaxIter',0,'NumLoc',100);
prevTurn = 1;
xResults = a(:)';
profitResults = zeros(1,n.F);
pastProfit = profitResults

while abs(change) > 0.1 & turn < 30
    for f = 1:n.F
        turn = turn+1;
        compIdx = 1:n.F*n.J;
        lowIdx = 1+n.J*(f-1);
        highIdx = n.J+n.J*(f-1);
        compIdx = setdiff(compIdx, [lowIdx:highIdx]);
        profEq = subs(profits,X(:,compIdx),pastX(:,compIdx));
        ghgEq = subs(swaGhg,X(:,compIdx),pastX(:,compIdx));
        focalProfEq = profEq(f);
        focalGhgEq = ghgEq(f);
        focalX = num2cell(X(:,lowIdx:highIdx));
        
        focalProf = matlabFunction(-focalProfEq,'Vars', {[focalX{:}]});
        focalGhg = matlabFunction(focalGhgEq,'Vars', {[focalX{:}]});
        focalGhgConst_ga = matlabFunction(focalGhgEq-ghgLim,[],'Vars', {[focalX{:}]},'Outputs', {'c','ceq'});
        x0 = {reshape(pastX(:,lowIdx:highIdx),1,[])};
%         focalProf(x0{:});
        focalGhg(x0{:});

        % optimize
        %min = fminunc(focalProf,x0{:}) % ,[],[])
%         focalGhgConst_Baron = matlabFunction(focalGhgEq-ghgLim,'Vars', {[focalX{:}]});
%         [min, negProfit] = baron(focalProf,[],[],[],lb,ub,focalGhgConst_Baron,-inf,0,cont,x0{:},opts);
        % min = fmincon(focalProf,x0{:},[],[],[],[],lb,ub,focalGhgConst_ga)
        % x = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon) 
        [min,negProfit,exitFlag,Output,population,scores] = ga(focalProf,n.J*2,[],[],[],[],lb,ub,focalGhgConst_ga) % issue constraining
        
        profit = -negProfit;
        pastX(:,lowIdx:highIdx) = reshape(min,2,[]);
        pastProfit(:,f) = profit;

        xResults(turn,:) = pastX(:)'
        profitResults(turn,:) = pastProfit
%         save times
    end
%     currentChange = results(turn,:)
    change = sum(xResults(turn,:)-xResults(prevTurn,:))
    prevTurn = turn
end
profitResults

%%

xResultsScaled = xResults
finalP = xResultsScaled(end,1:2:end)
finalX = xResultsScaled(end,2:2:end)
%xResultsScaled(:, [1,3]) = xResults(:, [1 3])
plot(xResultsScaled)
hold on
plot(profitResults,'x-')
finalDemands = subs(demands,X(:),xResults(end,:)')
finalUtilities = mean(subs(utilities(:,:),X(:),xResultsScaled(end,:)')) % this cant be right
finalGhg = subs(swaGhg,X(:),xResultsScaled(end,:)')
finalCost = subs(C,X(:),xResultsScaled(end,:)')
