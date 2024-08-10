function adj=adjoint_mat6x6(screw)

adj         =zeros(6,6);
adj(1:3,1:3)=skew_sym(screw(1:3));
adj(4:6,1:3)=skew_sym(screw(4:6));
adj(4:6,4:6)=skew_sym(screw(1:3));

% eof
