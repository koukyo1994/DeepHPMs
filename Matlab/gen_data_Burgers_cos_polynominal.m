%% Burgers equation and chaos
nn = 256;
steps = 200;

dom = [-8 8]; x = chebfun('x',dom); tspan = linspace(0,10,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) + 0.1*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = 0.1 + 1.6 * cos(pi * x / 8) - 0.4 * cos(pi * x / 4) - 0.1 * cos(pi * x / 2) - 0.01 * cos(pi * x);
u = spin(S,nn,1e-4);

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-8,8,nn+1);
x = x(1:end-1);
t = tspan;
pcolor(t,x,real(usol)); shading interp, axis tight, colormap(jet);
save('../MyData/burgers_cos_polynominal.mat','t','x','usol')


