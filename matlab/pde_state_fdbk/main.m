% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author
clc
clear all;
close all;

sys.ctrl_on = 1;
sys.second_order = 1;
sys.RGB_color1 = [1 0 0];
sys.RGB_color2 = [0 0 1];
sys.RGB_color3 = [0 1 0];
sys.style1 = '-';
sys.style2 = '-.';
sys.style3 = '--';


if (sys.ctrl_on == 1)
    sys.simH    = 5;
    sys.N       = 200;
else
    sys.simH    = 5;
    sys.N       = 60;
end;

% Simulation specific
sys.N_grid  = sys.N + 2;
sys.h       = 0.01;
sys.Tspan   = 0:sys.h:sys.simH;
 
sys.Delta = 1 / (sys.N + 1);
sys.xspan = (0:sys.Delta:1)';
sys.xspanT = (sys.Delta:sys.Delta:(1 - sys.Delta))';

sys.intArr = sys.Delta * [0.5; ones(sys.N, 1); 0.5];

sys.mu = 0.75;

sys.theta = 0.5 + 0.5 * exp(-sys.xspan) .* cosh(pi * sys.xspan);


sys.theta_L = -100;
sys.theta_U =  100;

sys.r_func = @(t)(1 + sin(2 * pi * t));


%% Tuning
sys.k_U = 100;

N = sys.N;


%% Control law
numIter = 30;

k_gain = zeros(sys.N_grid, 1);
for k = 1:numIter
    conv_k_theta = zeros(sys.N_grid, 1);

    for i = 2:sys.N_grid
        for j = 1:i
            conv_k_theta(i) = conv_k_theta(i) + ...
                sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == i)) * sys.theta(j) * k_gain(i - j + 1);
        end;
    end;
    k_gain = (conv_k_theta - sys.theta) / sys.mu;
end;



sys.k_gain = k_gain;

%% Initial conditions
u_sf_0 = sys.xspanT;
u_of_0 = sys.xspanT;
u_hat_0 = zeros(N, 1);
u_tr_0 = sys.xspanT;

U_sf_f_0 = 0;
U_of_f_0 = 0;
U_tr_f_0 = 0;


%% Initial condition
x0 = [u_sf_0; u_of_0; u_hat_0; u_tr_0; U_sf_f_0; U_of_f_0; U_tr_f_0];
% load('state');

%% Simulate
tic;
[t_log, x_log] = ode45(@(t, x) ode_solver(t, x, sys), sys.Tspan, x0);
toc;


%% Post-processing
numT = length(t_log);

xx1 = reshape(x_log(:, 1:(4*sys.N)), numT, sys.N, 4);
xx1_a = zeros(numT, sys.N + 2, 4);
xx1_a(:, 1, :) = 2*xx1(:, 1, :) - xx1(:, 2, :);
xx1_a(:, sys.N+2, :) = 2*xx1(:, sys.N, :) - xx1(:, sys.N-1, :);
xx1_a(:, 2:(sys.N+1), :) = xx1;


u_sf_a  = squeeze(xx1_a(:, :, 1));        u_sf = u_sf_a(:, 2:(sys.N+1));
u_of_a  = squeeze(xx1_a(:, :, 2));        u_of = u_of_a(:, 2:(sys.N+1));
u_hat_a = squeeze(xx1_a(:, :, 3));        u_hat = u_hat_a(:, 2:(sys.N+1));
u_tr_a = squeeze(xx1_a(:, :, 4));         u_tr = u_tr_a(:, 2:(sys.N+1));

numT = length(t_log);

U_sf = zeros(numT, 1);
U_of = zeros(numT, 1);
U_tr = zeros(numT, 1);

if (sys.ctrl_on == 1)
    for k = 1:numT
        U_sf(k) = sys.intArr' * (flipud(sys.k_gain) .* u_sf_a(k, :)'); 
        U_of(k) = sys.intArr' * (flipud(sys.k_gain) .* u_hat_a(k, :)');
        U_tr(k) = sys.intArr' * (flipud(sys.k_gain) .* u_tr_a(k, :)') + sys.r_func(t_log(k) + 1 / sys.mu);
    end;
end;

u_sf_norm = sqrt(u_sf_a.^2 * sys.intArr);
u_of_norm = sqrt(u_of_a.^2 * sys.intArr);
e_of_norm = sqrt((u_of_a - u_hat_a).^2 * sys.intArr);
u_tr_norm = sqrt(u_tr_a.^2 * sys.intArr);

%% Plotting
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);

%%% SYSTEM STATES
    % STATE FEEDBACK
    figure(1);
    subplot(131)
    surf(T_arr, X_arr, u_sf_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('$v$', 'interpreter','latex');
    title('State Feedback')
    view([25 36]);
    colormap(parula);
    
    % OUTPUT FEEDBACK
    subplot(132)
    surf(T_arr, X_arr, u_of_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    title('Output Feedback')
    zlabel('$v$', 'interpreter','latex');
    view([25 36]);
    colormap(parula);
    
    % ESTIMATION ERROR FEEDBACK
    subplot(133)
    surf(T_arr, X_arr, u_of_a - u_hat_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    title('Estimation Error Feedback')
    zlabel('$v - \hat v$', 'interpreter','latex');
    view([25 36]);
    colormap(parula);
    
    % TRACKING STATE
    figure(2);
    subplot(131);
    surf(T_arr, X_arr, u_tr_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('$v$', 'interpreter','latex');
    title('State Tracking')
    view([25 36]);
    colormap(parula);
    
    
    % STATES NORMS
    
    % STATE FEEDBACK
    subplot(132);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_sf_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||v||$', 'interpreter','latex');
    title("Norm of State Feedback")
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(u_sf_norm) - min(u_sf_norm));
    if (d > 0)
        V(3) = min(u_sf_norm) - 0.2*d;
        V(4) = max(u_sf_norm) + 0.2*d;
        axis(V);
    end;
    
    % OUTPUT FEEDBACK
    subplot(133);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_of_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||v||$', 'interpreter','latex');
    title("Norm of Output Feedback")
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(u_of_norm) - min(u_of_norm));
    if (d > 0)
        V(3) = min(u_of_norm) - 0.2*d;
        V(4) = max(u_of_norm) + 0.2*d;
        axis(V);
    end;
    
    
    %% ESTIMATION ERROR
    figure(3);
    subplot(131)
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, e_of_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||v - \hat v||$', 'interpreter','latex');
    title('Norm of Estimation Error')
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(e_of_norm) - min(e_of_norm));
    if (d > 0)
        V(3) = min(e_of_norm) - 0.2*d;
        V(4) = max(e_of_norm) + 0.2*d;
        axis(V);
    end;
    
    % TRACKING STATE NORM
    subplot(132)
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_tr_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||v||$', 'interpreter','latex');
    title('Norm of State Tracking')
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [0; u_tr_norm];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    %
%  COMBINED PLOT
    subplot(133)
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_sf_norm, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, u_of_norm, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, u_tr_norm, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||v||$', 'interpreter','latex');
    title("Combo plot")
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [u_sf_norm; u_of_norm; u_of_norm];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

    
    
    
    
%% Actuation
    figure(4);
    subplot(131)
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, U_sf, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, U_of, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, U_tr, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$U$', 'interpreter','latex');
    title("Controllers")
    legend(["Setpoint", "State feedback controller", "Output feedback controller", "Tracking controller"])
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [U_sf; U_of; U_tr];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
        
    % Objective
    subplot(132)
    plot(t_log, sys.r_func(t_log), 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_tr_a(:, 1), sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\vartheta$ and $r$', 'interpreter','latex');
    title("Objective")
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = u_tr_a(:, 1);
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    % Controller gain
    subplot(133)
    plot(sys.xspan, sys.k_gain, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    xlabel('x', 'interpreter','latex');
    ylabel('Gain $k$', 'interpreter','latex');
    title("Controller Gain")
    V = axis; V(2) = 1; axis(V);
    dataBlock = sys.k_gain;
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;