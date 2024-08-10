function coAdj=coAdjoint_mat6x6(config)

coAdj         =zeros(6,6);
coAdj(1:3,1:3)=config(1:3,1:3);
coAdj(1:3,4:6)=skew_sym(config(1:3,4))*config(1:3,1:3);
coAdj(4:6,4:6)=config(1:3,1:3);

% eof
