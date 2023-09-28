function status = save_ode_data(t, y, flag)
%SAVE_ODE_DATA Summary of this function goes here
%   Detailed explanation goes here

persistent y_record;
persistent t_record;

switch flag
    case 'init'
        t_record(end+1) = t(1);
        y_record(:,end+1) = y;
        results = struct('t',t_record, 'y', y_record);
        save('ode_intermediate_results.mat', 'results');
    case 'done'
        save('ode_intermediate_results.mat', 'results');
    otherwise
        t_record(end+1) = t;
        y_record(:,end+1) = y;
        results = struct('t',t_record, 'y', y_record);
        save('ode_intermediate_results.mat', 'results');
end

status = 0;

end

