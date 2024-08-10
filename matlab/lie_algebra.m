function se3=lie_algebra(screw)

se3         =zeros(4,4);
se3(1:3,1:3)=skew_sym(screw(1:3));
se3(1:3,4)  =screw(4:6);

% eof
