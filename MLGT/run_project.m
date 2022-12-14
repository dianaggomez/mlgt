
clear
% rng('default')

% a = [50 0]' % 1 veh
% a = [50 0; 50 0]' % 2 veh
% a = [50 0; 50 0; 50 0]' % 3 veh
% %a = [50 0.2; 50 0.2; 50 0.2; 50 0.2]' % 3 veh
% ghgLim = 50;
% isBev = 1;
% n.F = 1;

%% UNREGULATED

%% 1 vehicle
% setup
clear
a = [50 0]' % 1 veh
ghgLim = 300;
isBev = 0;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_1veh_Unc_100draws_v2")

%% 2 vehicle
clear
a = [50 0; 50 0]' % 2 veh
ghgLim = 300;
isBev = 0;
n.F = 1;
draws = 100

runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_2veh_Unc_100draws_v2")

%% analyze
figure
hist(max(results.P'))
figure
hist(min(results.P'))
figure
hist(median(results.P'))

%% 2 ICE + 1 BEV
clear
a = [50 0; 50 0; 50 0]' % 3 veh
ghgLim = 300;
isBev = 1;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_bev_Unc_100draws_v2")

%% REGULATED

%% 1 vehicle
% setup
clear
a = [50 0]' % 1 veh
ghgLim = 192;
isBev = 0;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_1veh_con192_100draws")

%% 2 vehicle
clear
a = [50 0; 50 0]' % 2 veh
ghgLim = 192;
isBev = 0;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_2veh_con192_100draws")

%% 2 ICE + 1 BEV + 192ghg
clear
a = [50 0; 50 0; 50 0]' % 3 veh
ghgLim = 192;
isBev = 1;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_bev_con192_100draws")

%% 2 ICE + 1 BEV + 100ghg
clear
a = [50 0; 50 0; 50 0]' % 3 veh
ghgLim = 100;
isBev = 1;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_bev_con100_100draws_v2")

%% 2 ICE + 1 BEV + 192ghg + BARON
clear
a = [50 0; 50 0; 50 0]' % 3 veh
ghgLim = 192;
isBev = 1;
n.F = 1;
draws = 100;
runs = 100;
[results] = collectData(a,ghgLim,isBev,n,runs,draws)
save("results_baron_bev_con192_100draws")

%% Load
% load("results_1veh_Unc_100draws") % 1 veh unc
% load("results_1veh_con192_100draws") % 1veh con
% 2 veh
% load("results_2veh_Unc_100draws")
% load("results_2veh_con192_100draws")
% BEV
% load("results_bev_Unc_100draws")    % BEV unc
load("results_bev_con192_100draws") % BEV real con
% load("results_bev_con100_100draws") % BEV strict
% load("results_baron_bev_con192_100draws") % baron

%% Plot
if isBev == 1
    xBev = results.X(:,1);
    pBev = results.P(:,1);
    [xHigh, highIdx] = max(results.X(:,2:3)');
    [xLow, lowIdx] = min(results.X(:,2:3)');
    xHigh = xHigh';
    xLow = xLow';
    [pHigh, highIdx] = max(results.P(:,2:3)'); % assuming eff vehicle is never lower price
    [pLow, lowIdx] = min(results.P(:,2:3)');
    pHigh = pHigh';
    pLow = pLow';    
%     [xLow, lowIdx] = min(results.X(:,2:3)');
%     xLow = xLow';
%     for i = 1:length(highIdx)
%         pHigh = results.P(i,highIdx+1)';
%         pLow = results.P(i,lowIdx+1)';
%     end
    X = [xLow, xHigh, xBev];
    P = [pLow, pHigh, pBev];
elseif width(a) == 2
    [xHigh, highIdx] = max(results.X(:,1:2)');
    [xLow, lowIdx] = min(results.X(:,1:2)');
    xHigh = xHigh';
    xLow = xLow';
    [pHigh, highIdx] = max(results.P(:,1:2)'); % assuming eff vehicle is never lower price
    [pLow, lowIdx] = min(results.P(:,1:2)');
    pHigh = pHigh';
    pLow = pLow'; 
    X = [xLow, xHigh];
    P = [pLow, pHigh];
elseif width(a) == 1
    xLow = results.X
    pLow = results.P    
    X = results.X
    P = results.P
end

figure  % tag: need to alter for 1 & 2 veh
plt = 1
row = 2
col = 2
subplot(row,col,plt)
    bin = 2
    histogram(pLow,BinWidth=bin)
    hold on
    if width(a) >1
        histogram(pHigh,BinWidth=bin)
    end
    if isBev == 1
        histogram(P(:,3),BinWidth=bin)
    end
    xlim([0 75])
    title('Price')
    xlabel('Price ($k)')
    ylabel('Count')
    legend('Low Eff Veh','High Eff Veh','Electric Veh')

% subplot(row,col,plt+1)  % note: instead plot demands
%     bin = 0.25
%     histogram(pLow,BinWidth=bin) % plot P low
%     hold on
%     if width(a) >1
%         histogram(pHigh,BinWidth=bin)
%     end
%     if isBev == 1
%         histogram(pBev,BinWidth=bin)
%     end
%     hold off
%     title('Price (with more bins)')
%     xlabel('Price ($k)')
%     ylabel('Count')
%     legend('Low Eff','High Eff','Electric')

plt = plt+1
subplot(row,col,plt) 
    bin = .1;    
    histogram(xLow,BinWidth=bin) % plot x
    hold on
    if width(a) >1
        histogram(xHigh,BinWidth=bin)
    end
    xlim([0 0.75])
    title('Efficiency of Gas Vehicles')
    legend('Low Eff Veh','High Eff Veh',Location='northwest')
    xlabel("% Emissions Reduction")
    ylabel('Count')
% plt = plt+1
% subplot(row,col,plt) % plot pi
%     histogram(results.Pi)
%     xlim([0 inf])
%     title('Agent Reward')
%     xlabel("Profit")
%     ylabel('Count')
% hold off
%     % legend()

% histogram(,BinWidth="1")
% hold on
% histogram(,BinWidth="1")
% histogram(,BinWidth="1")
profitibility = results.demand.*(results.P-results.Cost); 
%% viz profits for low and high (single firm unc)
% paste low & high from excel
plt = plt+1
subplot(row,col,plt)  % plot pi

if isBev == 1
    idx = any(results.X(:,2:3) < 0.3,2)
else
    idx = any(results.X < 0.01,2)
end

lowPi = results.Pi(idx)
highPi = results.Pi(~idx)

hold on

histogram(highPi,BinWidth=bin)
histogram(lowPi,BinWidth=bin)
legend("Upgrades","No Upgrades")
title('Agent Reward')
xlabel("Profit")
ylabel('Count')

%% Plot Draws Distributions
% figure
% subplot(2,1,1)
plt = plt+1
subplot(row,col,plt)  % plot pi

lowDraws = results.Betas(idx,:)
meanLow = mean(lowDraws')
highDraws = results.Betas(~idx,:)
meanHigh = mean(highDraws')
% % plot histogram across all samples
% histogram(lowDraws)
% hold on
% histogram(highDraws)
% title('Distribution Across All Samples')
% meanLow1 = mean(meanLow)
% meanHigh1 = mean(meanHigh)
% xline(meanLow1,'b--')
% xline(meanHigh1,'r-')
% legend('No Upgrades Draws','Upgrades Draws','No Upgrades Mean','No Upgrades Mean')
% xlabel('wtp for opCost')

% mean of samples
% subplot(2,1,2)
bin = 0.05
% figure

hold on
histogram(meanHigh,BinWidth=bin)
histogram(meanLow,BinWidth=bin)

title('Mean Draws of Each Sample')
meanLow = mean(meanLow)
meanHigh = mean(meanHigh)
xline(meanLow,'r--')
xline(meanHigh,'b-')
legend('Upgrades Draws Mean','No Upgrades Draws Mean') %,'No Upgrades Mean','No Upgrades Mean')
xlabel('wtp for opCost')
ylim([0 40])

% note: low tech appears to have less negative pref for op cost
% takeaway: difference driven by difference in pref