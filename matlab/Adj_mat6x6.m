function Adj=Adj_mat6x6(config)

Adj         =zeros(6,6);
Adj(1:3,1:3)=config(1:3,1:3);
Adj(4:6,1:3)=skew_sym(config(1:3,4))*config(1:3,1:3);
Adj(4:6,4:6)=config(1:3,1:3);
