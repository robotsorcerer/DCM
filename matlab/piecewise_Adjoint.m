function Adjg=piecewise_Adjoint(x,theta,xci)

adjxci       =adjoint_mat6x6(xci);

if theta==0
    Adjg        =diag([1 1 1 1 1 1])+x*adjxci;
else
    Adjg        =diag([1 1 1 1 1 1])+((3*sin(x*theta)-x*theta*cos(x*theta))/(2*theta))*adjxci+...
                 ((4-4*cos(x*theta)-x*theta*sin(x*theta))/(2*theta^2))*adjxci^2+...
                 ((sin(x*theta)-x*theta*cos(x*theta))/(2*theta^3))*adjxci^3+...
                 ((2-2*cos(x*theta)-x*theta*sin(x*theta))/(2*theta^4))*adjxci^4;
end

% eof
