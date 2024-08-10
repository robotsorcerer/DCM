close all 

r=5; l=5; nTheta=100, nL = 20;

theta = linspace(0,2*pi,nTheta+1);
x = r * cos(theta);
y = r * sin(theta);

z = linspace(0,l,nL)';
xshift = repmat( sin(z), 1, nTheta+1); %this is a function of z

X = repmat(x,nL,1);% + xshift;
Y = repmat(y,nL,1) + xshift;
Z = repmat(z, 1, nTheta+1);

% Plot
surf(X,Y,Z)