% This script uses kernel principal component analysis for fault
% detection in an academic example. Hoffman's implementation of KPCA is
% used with minimal modification.
%
% © Rushikesh Kamalapurkar
%
clear all; close all; clc;
addpath('../../lib')

%% Initialization
dyn1 = @(t,x) -x + [x(2)*sin(pi/2*x(1)); x(1)*cos(pi/2*x(2))];
dyn2 = @(t,x) -x + [0.9*x(2)*sin(pi/5*x(1)); 0.8*x(1)*cos(pi/3*x(2))];
n = 2; % State dimension
tSpan = 0:0.01:2; % Time span
M = [50,100,150]; % # Training trajectories
faultyTestM = 20; % # Faulty test trajectories 
normalTestM = 20; % # Normal test trajectories
mu = 5; % Kernel width
N = 20; % Number of eigenvectors
measNoiseSD = 0.01; % Standard deviation of added measurement noise
sampNoiseRange = 0; % Range of added sampling rate noise
trials = 100; % Number of repeated trials
thresholdMultiplier = 2; % Threshold = max training error times this
% Storage matrices

normalTestInitialParams = zeros(trials,normalTestM,numel(M));
faultyTestInitialParams = zeros(trials,faultyTestM,numel(M));
RTest = zeros(normalTestM+faultyTestM,trials,numel(M));
if numel(M)==1
    RTrain = zeros(M,trials);
    trainInitialParams = zeros(trials,M);
else
    RTrain = cell(numel(M),1);
    trainInitialParams = cell(numel(M),1);
    for ii=1:numel(M)
        RTrain{ii,1} = zeros(M(ii),trials);
        trainInitialParams{ii,1} = zeros(trials,M(ii));
    end
end

%% Monte-carlo trials
for ii=1:numel(M)
for trial = 1:trials
    % Training data
    trainInitialParam = 2*pi*rand(1,M(ii));
    trainX0 = [sin(trainInitialParam);cos(trainInitialParam)];
    % Generate training data
    trainPaths = zeros(length(trainX0),n*length(tSpan));
    for i = 1:length(trainX0)
        noisytSpan = tSpan + sampNoiseRange*2*(rand(size(tSpan))-0.5);
        [~,temp] = ode45(dyn1,noisytSpan,trainX0(:,i));
        temp = temp.';
        trainPaths(i,:)=temp(:)';
    end
    trainPaths = trainPaths + measNoiseSD*randn(size(trainPaths));
    
    % Faulty test data
    faultyTestInitialParam = 2*pi*rand(1,faultyTestM);
    faultyTestX0 = [sin(faultyTestInitialParam);cos(faultyTestInitialParam)];
    % Generate faulty test data
    faultyTestPaths = zeros(length(faultyTestX0),n*length(tSpan));
    for i = 1:length(faultyTestX0)
        noisytSpan = tSpan + sampNoiseRange*2*(rand(size(tSpan))-0.5);
        [~,temp] = ode45(dyn2,noisytSpan,faultyTestX0(:,i));
        temp = temp.';
        faultyTestPaths(i,:)=temp(:)';
    end
    faultyTestPaths = faultyTestPaths + measNoiseSD*randn(size(faultyTestPaths));
    
    % Normal test data
    normalTestInitialParam = 2*pi*rand(1,normalTestM);
    normalTestX0 = [sin(normalTestInitialParam);cos(normalTestInitialParam)];
    % Generate Normal test data
    normalTestPaths = zeros(length(normalTestX0),n*length(tSpan));
    for i = 1:length(normalTestX0)
        noisytSpan = tSpan + sampNoiseRange*2*(rand(size(tSpan))-0.5);
        [~,temp] = ode45(dyn1,noisytSpan,normalTestX0(:,i));
        temp = temp.';
        normalTestPaths(i,:)=temp(:)';
    end
    normalTestPaths = normalTestPaths + measNoiseSD*randn(size(normalTestPaths));

    if numel(M) == 1
        trainInitialParams(trial,:) = trainInitialParam;
    else
        trainInitialParams{ii}(trial,:) = trainInitialParam;
    end
    faultyTestInitialParams(trial,:) = faultyTestInitialParam;
    normalTestInitialParams(trial,:) = normalTestInitialParam;
    
    % KPCA
    [rtrain,rtest] = kpcabound(trainPaths,mu,N,[normalTestPaths;faultyTestPaths]);
    
    % Store reconstruction errors
    RTest(:,trial,ii) = rtest;
    if numel(M) == 1
        RTrain(:,trial) = rtrain;
    else
        RTrain{ii}(:,trial) = rtrain;
    end
end
end
%% Fault detection
if numel(M) == 1
    % Threshold for fault detection for each trial
    epsilon = thresholdMultiplier*max(RTrain);
else
    epsilon = zeros(1,trials,numel(M));
    for ii=1:numel(M)
        epsilon(1,:,ii) = thresholdMultiplier*max(RTrain{ii});
    end
end

% Total false positives per trial
falsePositives = sum(RTest(1:normalTestM,:,:) > epsilon);

% Total false negatives per trial
falseNegatives = sum(RTest(normalTestM+1:normalTestM+faultyTestM,:,:) < epsilon);

% Mixed points over per trial
mixedPoints = sum(RTest(normalTestM+1:normalTestM+faultyTestM,:,:) < max(RTest(1:normalTestM,:,:)))...
    + sum(RTest(1:normalTestM,:,:) > min(RTest(normalTestM+1:normalTestM+faultyTestM,:,:)));

disp(['False positive percentage = ' num2str(sum(falsePositives)*100/(trials*(normalTestM+faultyTestM)))])
disp(['False negative percentage = ' num2str(sum(falseNegatives)*100/(trials*(normalTestM+faultyTestM)))])
disp(['Mixing percentage = ' num2str(sum(mixedPoints)*100/(trials*(normalTestM+faultyTestM)))])

%% Plots
if numel(M) == 1
    scatter(1:normalTestM,log(RTest(1,1:normalTestM)),'b','filled')
    hold on
    scatter(1:normalTestM,log(RTest(1,normalTestM+1:normalTestM+faultyTestM)),'r','filled')
    line([1,20],log([2*max(RTrain(1,:)),2*max(RTrain(1,:))]),'color','g','linewidth',2)
end