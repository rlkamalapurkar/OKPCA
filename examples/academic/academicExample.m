% This script uses occupation kernel principal component analysis for fault
% detection in an academic example.
%
% © Rushikesh Kamalapurkar
%
clear all; close all; clc;
addpath('../../lib')
%% Initialization
% Nominal system
n = 2; % State dimension
dyn1 = @(t,x) -x + [x(2)*sin(pi/2*x(1)); x(1)*cos(pi/2*x(2))];

% Faulty system
dyn2 = @(t,x) -x + [0.9*x(2)*sin(pi/5*x(1)); 0.8*x(1)*cos(pi/3*x(2))];

% Dataset parameters
tf = 2;
h = 0.01;
tspan = 0:h:tf; % Time span
M = 100;%[50,100,150]; % # Training trajectories
normalTestM = 20; % Number of normal test trajectories 
faultyTestM = 20; % Number of faulty test trajectories 
measNoiseSD = 0.01; % Standard deviation of added measurement noise
sampNoiseRange = 0; % Range of added sampling rate noise

% Kernel parameters
mu = 0.6; % Kernel width
k = KernelRKHS('Gaussian',mu); % Kernel

% PCA parameters
N = 20; % Number of eigenvectors

% Fault detection parameters
thresholdMultiplier = 2; % Threshold = max training error times this

% MCMC Parameters
trials = 100; % Number of trials

% Matrices to store results
RTEST = zeros(normalTestM+faultyTestM,trials,numel(M));
normalTestInitialParams = zeros(trials,normalTestM,numel(M));
faultyTestInitialParams = zeros(trials,faultyTestM,numel(M));
if numel(M)==1
    RTRAIN = zeros(M,trials);
    trainInitialParams = zeros(trials,M);
else
    RTRAIN = cell(numel(M),1);
    trainInitialParams = cell(numel(M),1);
    for ii=1:numel(M)
        RTRAIN{ii,1} = zeros(M(ii),trials);
        trainInitialParams{ii,1} = zeros(trials,M(ii));
    end
end

%% Repeated trials
for ii = 1:numel(M)
    for trial = 1:trials
        % Initial states for training data
        trainInitialParam = 2*pi*rand(1,M(ii));
        trainX0 = [sin(trainInitialParam);cos(trainInitialParam)];
        % Generate training data
        trainPaths = zeros(n,length(tspan),length(trainX0));
        tTrain = zeros(length(tspan),length(trainX0));
        for i = 1:length(trainX0)
            noisytSpan = tspan + sampNoiseRange*2*(rand(size(tspan))-0.5);
            [~,temp] = ode45(dyn1,noisytSpan,trainX0(:,i));
            trainPaths(:,:,i)=temp';
            tTrain(:,i) = noisytSpan;
        end
        
        % Initial states for normal test data
        normalTestInitialParam = 2*pi*rand(1,normalTestM);
        normalTestX0 = [sin(normalTestInitialParam);cos(normalTestInitialParam)];
        % Generate normal test data
        normalTestPaths = zeros(n,length(tspan),length(normalTestX0));
        tNormalTest = zeros(length(tspan),length(normalTestX0));
        for i = 1:length(normalTestX0)
            noisytSpan = tspan + sampNoiseRange*2*(rand(size(tspan))-0.5);
            [~,temp] = ode45(dyn1,noisytSpan,normalTestX0(:,i));
            normalTestPaths(:,:,i)=temp';
            tNormalTest(:,i) = noisytSpan;
        end
        
        % Initial states for abnormal test data
        faultyTestInitialParam = 2*pi*rand(1,faultyTestM);
        faultyTestX0 = [sin(faultyTestInitialParam);cos(faultyTestInitialParam)];
        % Generate abnormal test data
        faultyTestPaths = zeros(n,length(tspan),length(faultyTestX0));
        tFaultyTest = zeros(length(tspan),length(faultyTestX0));
        for i = 1:length(faultyTestX0)
            noisytSpan = tspan + sampNoiseRange*2*(rand(size(tspan))-0.5);
            [~,temp] = ode45(dyn2,noisytSpan,faultyTestX0(:,i));
            faultyTestPaths(:,:,i)=temp';
            tFaultyTest(:,i) = noisytSpan;
        end
        
        % Add noise
        trainPaths = trainPaths + measNoiseSD*randn(size(trainPaths));
        normalTestPaths = normalTestPaths + measNoiseSD*randn(size(normalTestPaths));
        faultyTestPaths = faultyTestPaths + measNoiseSD*randn(size(faultyTestPaths));
        
        % Store initial conditions
        if numel(M) == 1
            trainInitialParams(trial,:) = trainInitialParam;
        else
            trainInitialParams{ii}(trial,:) = trainInitialParam;
        end
        normalTestInitialParams(trial,:,ii) = normalTestInitialParam;
        faultyTestInitialParams(trial,:,ii) = faultyTestInitialParam;

        % OKPCA Reconstruction Error
        [RTest,RTrain] = OKPCAReconstructionError(k,trainPaths,tTrain,cat(3,normalTestPaths,faultyTestPaths),cat(2,tNormalTest,tFaultyTest),N);
        
        % Store reconstruction errors
        RTEST(:,trial,ii) = RTest;
        if numel(M) == 1
            RTRAIN(:,trial) = RTrain;
        else
            RTRAIN{ii}(:,trial) = RTrain;
        end
    end
end

%% Fault detection
if numel(M) == 1
    % Threshold for fault detection for each trial
    epsilon = thresholdMultiplier*max(RTRAIN);
else
    epsilon = zeros(1,trials,numel(M));
    for ii=1:numel(M)
        epsilon(1,:,ii) = thresholdMultiplier*max(RTRAIN{ii});
    end
end

% Total false positives per trial
falsePositives = sum(RTEST(1:normalTestM,:,:) > epsilon);

% Total false negatives per trial
falseNegatives = sum(RTEST(normalTestM+1:normalTestM+faultyTestM,:,:) < epsilon);

% Mixed points over per trial
mixedPoints = sum(RTEST(normalTestM+1:normalTestM+faultyTestM,:,:) < max(RTEST(1:normalTestM,:,:)))...
    + sum(RTEST(1:normalTestM,:,:) > min(RTEST(normalTestM+1:normalTestM+faultyTestM,:,:)));

disp(['False positive percentage = ' num2str(sum(falsePositives)*100/(trials*(normalTestM+faultyTestM)))])
disp(['False negative percentage = ' num2str(sum(falseNegatives)*100/(trials*(normalTestM+faultyTestM)))])
disp(['Mixing percentage = ' num2str(sum(mixedPoints)*100/(trials*(normalTestM+faultyTestM)))])
%% Plots
if numel(M) == 1
    [~,bestTrial] = min(mixedPoints);
    scatter(1:normalTestM,log(RTEST(1:normalTestM,bestTrial)),'b','filled');
    hold on
    scatter(1:normalTestM,log(RTEST(normalTestM+1:normalTestM+faultyTestM,bestTrial)),'r','filled');
    line([1,normalTestM],[log(epsilon(bestTrial)),log(epsilon(bestTrial))],'color','g','linewidth',2);
    set(gca,'fontsize',14);
    legend("Normal","Faulty","Threshold",'interpreter','latex','fontsize',14);
    xlabel("Test point",'interpreter','latex','fontsize',14);
    ylabel("Log reconstruction error",'interpreter','latex','fontsize',14);
    xlim([1 20]);
    hold off
    temp = [(1:normalTestM).',...
        log(RTEST(1:normalTestM,bestTrial)),...
        log(RTEST(normalTestM+1:normalTestM+faultyTestM,bestTrial)),...
        log(epsilon(bestTrial))*ones(normalTestM,1)];
    save('Exp1BestTrial.dat','temp','-ascii');
    
    figure
    [~,worstTrial] = max(mixedPoints);
    scatter(1:normalTestM,log(RTEST(1:normalTestM,worstTrial)),'b','filled');
    hold on
    scatter(1:normalTestM,log(RTEST(normalTestM+1:normalTestM+faultyTestM,worstTrial)),'r','filled');
    line([1,normalTestM],[log(epsilon(worstTrial)),log(epsilon(worstTrial))],'color','g','linewidth',2);
    set(gca,'fontsize',14);
    legend("Normal","Faulty","Threshold",'interpreter','latex','fontsize',14);
    xlabel("Test point",'interpreter','latex','fontsize',14);
    ylabel("Log reconstruction error",'interpreter','latex','fontsize',14);
    xlim([1 20]);
    hold off
    temp = [(1:normalTestM).',...
        log(RTEST(1:normalTestM,worstTrial)),...
        log(RTEST(normalTestM+1:normalTestM+faultyTestM,worstTrial)),...
        log(epsilon(worstTrial))*ones(normalTestM,1)];
    save('Exp1WorstTrial.dat','temp','-ascii');
    
    figure
    hold on
    handle1 = plot(squeeze(trainPaths(1,:,:)),squeeze(trainPaths(2,:,:)),'g');
    handle2 = plot(squeeze(normalTestPaths(1,:,:)),squeeze(normalTestPaths(2,:,:)),'b');
    handle3 = plot(squeeze(faultyTestPaths(1,:,:)),squeeze(faultyTestPaths(2,:,:)),'r');
    set(gca,'fontsize',14);
    legend([handle1(1),handle2(1),handle3(1)],'Training Data','Normal Test Data','Faulty Test Data','interpreter','latex','fontsize',14)
    xlabel("$x_1$",'interpreter','latex','fontsize',14);
    ylabel("$x_2$",'interpreter','latex','fontsize',14);
    hold off
    temp = [squeeze(trainPaths(1,:,:)), squeeze(trainPaths(2,:,:))];
    save('Exp1TrainNoisy.dat','temp','-ascii');
    temp = [squeeze(normalTestPaths(1,:,:)),squeeze(normalTestPaths(2,:,:))];
    save('NormalExp1TrainNoisy.dat','temp','-ascii');
    temp = [squeeze(faultyTestPaths(1,:,:)),squeeze(faultyTestPaths(2,:,:))];
    save('FaultyExp1TrainNoisy.dat','temp','-ascii');
end