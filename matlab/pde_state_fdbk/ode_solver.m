% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_solver(t, x, sys)

    N = sys.N;
    N_grid = sys.N_grid;
    
% Extraction of variables
    dummy1 = reshape(x(1:(4*N)), N, 4);
    dummy1_a = [2*dummy1(1, :) - dummy1(2, :); dummy1; 2*dummy1(N, :) - dummy1(N-1, :)];
    
    u_sf_a  = dummy1_a(:, 1);        %u_sf = u_sf_a(2:(N_grid-1));
    u_of_a  = dummy1_a(:, 2);        %u_of = u_of_a(2:(N_grid-1));
    u_hat_a = dummy1_a(:, 3);        %u_hat = u_hat_a(2:(N_grid-1));
    u_tr_a  = dummy1_a(:, 4);        %u_hat = u_hat_a(2:(N_grid-1));
    
    U_sf_f = x(4*N + 1);
    U_of_f = x(4*N + 2);
    U_tr_f = x(4*N + 3);
    
    
%% Control law
    if (sys.ctrl_on == 1)
        U_sf = sys.intArr' * (flipud(sys.k_gain) .* u_sf_a);
        U_of = sys.intArr' * (flipud(sys.k_gain) .* u_hat_a);
        U_tr = sys.intArr' * (flipud(sys.k_gain) .* u_tr_a) + sys.r_func(t + 1 / sys.mu);
    else
        U_sf = 0;
        U_of = 0;
        U_tr = 0;
    end;
    
    
%% State augmentation
    u_sf_a(sys.N_grid)  = U_sf;
    u_of_a(sys.N_grid)  = U_of;
    u_hat_a(sys.N_grid) = U_of;
    u_tr_a(sys.N_grid)  = U_tr;
    
    
%% Spatial derivatives
    if (sys.second_order == 1)
        u_sf_x         = [(- 3*u_sf_a(2:(N_grid-2)) + 4 * u_sf_a(3:(N_grid-1)) - u_sf_a(4:N_grid)) / (2 * sys.Delta); (u_sf_a(N_grid) - u_sf_a(N_grid-1)) / sys.Delta];
        u_of_x         = [(- 3*u_of_a(2:(N_grid-2)) + 4 * u_of_a(3:(N_grid-1)) - u_of_a(4:N_grid)) / (2 * sys.Delta); (u_of_a(N_grid) - u_of_a(N_grid-1)) / sys.Delta];
        u_hat_x        = [(- 3*u_hat_a(2:(N_grid-2)) + 4 * u_hat_a(3:(N_grid-1)) - u_hat_a(4:N_grid)) / (2 * sys.Delta); (u_hat_a(N_grid) - u_hat_a(N_grid-1)) / sys.Delta];
        u_tr_x         = [(- 3*u_tr_a(2:(N_grid-2)) + 4 * u_tr_a(3:(N_grid-1)) - u_tr_a(4:N_grid)) / (2 * sys.Delta); (u_tr_a(N_grid) - u_tr_a(N_grid-1)) / sys.Delta];
    else
        u_sf_x         = (u_sf_a(3:N_grid) - u_sf_a(2:(N_grid-1))) / (sys.Delta);
        u_of_x         = (u_of_a(3:N_grid) - u_of_a(2:(N_grid-1))) / (sys.Delta);
        u_hat_x        = (u_hat_a(3:N_grid) - u_hat_a(2:(N_grid-1))) / (sys.Delta);
        u_tr_x         = (u_tr_a(3:N_grid) - u_tr_a(2:(N_grid-1))) / (sys.Delta);
    end;

    
%% Dynamics
    u_sf_t  = sys.mu * u_sf_x  + sys.theta(2:(N_grid-1)) * u_sf_a(1);
    u_of_t  = sys.mu * u_of_x  + sys.theta(2:(N_grid-1)) * u_of_a(1);
    u_hat_t = sys.mu * u_hat_x + sys.theta(2:(N_grid-1)) * u_of_a(1);
    u_tr_t  = sys.mu * u_tr_x  + sys.theta(2:(N_grid-1)) * u_tr_a(1);
    
    U_sf_f_t = sys.k_U * (U_sf - U_sf_f);
    U_of_f_t = sys.k_U * (U_of - U_of_f);
    U_tr_f_t = sys.k_U * (U_tr - U_tr_f);

    dt = [u_sf_t; u_of_t; u_hat_t; u_tr_t; U_sf_f_t; U_of_f_t; U_tr_f_t];