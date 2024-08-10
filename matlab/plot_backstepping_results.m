%% load and plot fixed point tracking data of the backstepping controller
clear;clc;
close all
load('ode_intermediate_results_fixed_point.mat');

time = results.t;
state = results.y;

% N: dimension of the joint state; M: length of trajectory. 
[N,M] = size(state);

len = 100;

figure;
colors = lines(N);
for i = 4:6
    % plot the linear strain states of the first piece
    plot(time(1:len+1), state(i,1:len+1), 'Color', colors(i,:), 'LineWidth', 5);
    hold on
end

% plot the constant (fixed point) tracking reference for the z-axis linear
% strain.
ref = [];
for kk = 1:len+1
    output = 0.5;
    ref = [ref output(end)];
end

%subplot(211)
plot(time(1:len+1), ref, 'r-.', 'LineWidth', 5);
legend('X', 'Y', 'Z', 'SP', 'FontSize', 10, 'FontWeight', 'bold',...
        'location', 'best');
title("Setpoint, $Z=0.5$.",'Interpreter','latex',...
           'FontSize', 28, 'FontWeight', 'bold');

ylim([-0.1 1.2]);
xlabel('Optimization steps (X 10K)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Linear strain', 'FontSize', 28, 'FontWeight', 'bold');
% 
%set(gca,'units','pix','pos',[100,100,400,400])
%export_fig('/home/lex/Documents/Papers/Pubs23/SoRoBC/figures/fixed_setpoint.eps',gca)
% saveas(gcf,'/home/lex/Documents/Papers/Pubs23/SoRoBC/figures/fixed_setpoint','epsc')

%% load and plot sinusoidal reference tracking data of the backstepping controller 

% clear;
load('ode_intermediate_results_sin_ref.mat')

time = results.t;
state = results.y;

% the sinusoidal reference signal is given by qd
delta = 0.5;
npie = 2;
qd = @(t) repmat([0;0;0;1;0; sin(2*t)*delta], npie, 1);
qd_dot = @(t) repmat([0;0;0;1;0; 2*cos(2*t)*delta], npie, 1);
qd_ddot = @(t) repmat([0;0;0;1;0; -4*sin(2*t)*delta], npie, 1);

% N: dimension of the joint state; M: length of trajectory. 
[N,M] = size(state);

% extract the valid data slice (somehow the simulation data from two
% experiments were concataenated). 
C = 5797;
len = 2259;

figure;
colors = lines(N);
for i = 4:6
    plot(time(1:len+1), state(i,C:C+len), 'Color', colors(i,:), 'LineWidth', 5);
    hold on
end

% plot the reference signal
ref = [];
for kk = 1:len+1
    output = qd(time(kk));
    ref = [ref output(end)];
end

%subplot(212)
plot(time(1:len+1), ref, 'r-.', 'LineWidth', 5);
legend('X', 'Y', 'Z', 'Ref.', 'FontSize', 28, 'FontWeight', 'bold');

xlim([0 2.2]);


title("Time-Varying Trajectory Tracking",...
           'FontSize', 28, 'FontWeight', 'bold');
xlabel('Optimization steps (X 10K)', 'FontSize', 28, 'FontWeight', 'bold');
ylabel('Linear strain', 'FontSize', 28, 'FontWeight', 'bold');

%saveas(gcf,'/home/lex/Documents/Papers/Pubs23/SoRoBC/figures/varying','epsc')



