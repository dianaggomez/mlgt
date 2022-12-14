function [results] = collectData(a,ghgLim,isBev,n,runs,draws)
tic
for i = 1:runs
    [d,p,x,pi,demand,utilities,ghg,cost,betas] = evolutionary(a,ghgLim,isBev,n,draws);
    results.D(i,:) = d;
    results.P(i,:) = p;
    results.X(i,:) = x;
    results.Pi(i,:) = pi;
    results.demand(i,:) = demand;
    results.utilities(i,:) = utilities;
    results.Ghg(i,:) = ghg;
    results.Cost(i,:) = cost;
    results.Betas(i,:) = betas';
    i
end
toc
end