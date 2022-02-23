function [eps, n, k]= getMoS2epsnk(lambda,Ef,Temp)
a = [2.0089e+05
   5.7534e+04
   8.1496e+04
   8.2293e+04
   3.3130e+05
   4.3906e+06
   1.0853e-02
   5.9099e-02
   1.1302e-01
   1.1957e-01
   2.8322e-01
   7.8515e-01
   2.3224e+01
   2.7982e+00
   3.0893e-01];
NX = 20;
Esa = [0 get_eigens_mos2(0,Temp,NX) 4.34];
b = a2b(a,Ef,Temp);           

eps = zeros(1,length(lambda));
for ii = 1:length(lambda)
    eps(ii) = getepsEfsingle(b,Esa,Ef,Temp,lambda(ii),6);
end
[n, k] = eps2nk(eps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function epsmos2 = getepsEfsingle(params,Esas,Ef,Temp,xdata,nn)
ehbar = 1.519250349719305e+15;
omegap = 4.6075e-03*ehbar;
fs = params(1:nn);
gs = params(1+nn:2*nn)*ehbar;
os = Esas*ehbar;

EeV = 1240e-9./xdata;
kbT = 8.621738e-5*Temp;
d =   exp(-(2*Ef-kbT).^2/2*300);
as = params(2*nn+1);
bs = params(2*nn+2);
cs = params(2*nn+3);
epsgi = as*exp(-(EeV-bs+kbT).^2/2/cs.^2);

oml = 1.883651567308853e+09*(xdata.^(-1));
epsmos2 = 4.44;
for mm = 1:nn
    epsmos2 = epsmos2 + fs(mm).*omegap^2.*(os(mm).^2-oml.^2-1i*gs(mm).*oml).^(-1);
end


epsmos2 = epsmos2+1i*epsgi;
EeVi = [linspace(-100.0,EeV-0.01,20000) linspace(EeV+0.01,100,20000)];
epsi = as*exp(-(EeVi-bs+kbT).^2/2/cs.^2);
epsr = trapz(EeVi,EeVi.*epsi./(EeVi.^2-EeV^2))/pi;

epsmos2 = epsmos2+epsr;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function b = a2b(a,Ef,Temp)
kbT = 8.621738e-5*Temp;
ddd =   exp(-(2*Ef-kbT).^2/(8.621738e-5*300));
b = a;
b(2) = b(2)*ddd^4;
b(3) = b(3)*ddd;
b(4) = b(4)*ddd;
b(5) = b(5)*ddd;
b(8) = b(8)/ddd;
b(9) = b(9)/ddd;
b(10) = b(10)/ddd;
b(11) = b(11)/ddd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ef = convertV2Ef(Vg)
hbar_eVs = 6.5821192815e-16;  % Reduced Planck Constant (unit: eV.s)
e_mass = 9.10938188e-31;    % electron mass (kg)
Cg = 1.2e-4;                                            % F/m^2
Ef = hbar_eVs^2*pi*Cg*(Vg+107)/(2*0.35*e_mass);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [n, k] = eps2nk(eps_mat)
r_mat = real(eps_mat);
i_mat = imag(eps_mat);
n = sqrt(0.5*(sqrt(r_mat.^2+i_mat.^2)+r_mat));
k = sqrt(0.5*(sqrt(r_mat.^2+i_mat.^2)-r_mat));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Esa = get_eigens_mos2(Ef,Temp,NX)
%__________________________ Univeral Constants __________________________
hbar = 1.05457172647e-34;           % Reduced Planck Constant (unit: J.s)
e_c = 1.60217656535e-19;            % Elementary Charge (unit: Coulombs)
c = 299792458;                      % speed of ligth (unit: m/s)
e_mass = 9.10938188e-31;            % electron mass (kg)
eps0 = 8.854187817e-12;             % permittivity of free space (F/m)
mu0 = 4e-7*pi;                      % permeability of free space (H/m)
a0 = 4*pi*eps0*hbar^2/e_mass/e_c^2; % Bohr radius (m)
Ry = 13.6056925330;                 % Rydberg energy (eV)
%__________________________ Variables __________________________________
m  = e_mass*0.32;                   % reduced mass (kg)
a_lattice0 = 0.316e-9;                % lattice constant
ESO = 0.152-Ef;                        % valence band split (eV)
epsr = 2.5;                           % Dielectric constant for MX2
E1sa = 1.886;
kT = 8.621738e-5*Temp;
a_lattice = a_lattice0;
a0 = a0/tanh(1.1/(8.621738e-5.*Temp).^0.25);
% Mesh Creation
[x, y] = meshgrid((-NX:NX)*a_lattice/2,((-NX-1:NX+1))*a_lattice*sqrt(3)/2);
[px, py] = size(x);
[xi, yi] = meshgrid(1:px,1:py);
A = ones(size(xi));
A(mod(xi+yi,2)==0)=0;
xs = x(:);      ys = y(:);      As = A(:);
xs(As==0)=[];   ys(As==0)=[];

asdq = find(sqrt(xs.^2+ys.^2)>a_lattice*NX/2);
xs(asdq) = [];          ys(asdq) = [];

% Initialization
nj = length(xs);                % number of points of the mesh
zeroj = find(xs==0 & ys==0);    % where the center element is

% Diagonal terms
H = blkdiag(diag(-2*a0*Ry/epsr./sqrt(xs.^2+ys.^2)));
% Singularity treatment
H(zeroj,zeroj)=-2*a0*Ry/epsr/a_lattice;     % H(zeroj,zeroj)=-V0

% Single Particle Contributions
for j = 1:nj
    rs = sqrt((xs(j)-xs).^2+(ys(j)-ys).^2);
    asd = find(rs<1.01*a_lattice);
    H(j,asd) = H(j,asd)+2*Ry/3*e_mass/m*(a0/a_lattice)^2+1i*ESO/9*sin(4*pi*(xs(j)-xs(asd).')/3/a_lattice);
end
H = H+4.861*eye(nj);

[eigenvectors, eigenvalues] = eig(H);
eigens0 = diag(eigenvalues);

shifted_eigs = round(1e3*diag(eigenvalues))/1e3;
[qwe, zxc] = unique(shifted_eigs([1:15]));
% disp('_______________ ALL STATES ________________________');
% disp(num2str([zxc shifted_eigs(zxc) 1240./shifted_eigs(zxc)]));
% disp('_________________S STATES_________________________');
PHI = eigenvectors.*conj(eigenvectors);
Esa = shifted_eigs(1);
for ix = 2:20
    [xcv, cvb] = max(PHI(:,ix));
    if cvb == zeroj          % s state
        A = [ix eigens0(ix) 1240./eigens0(ix)];
        % disp(num2str(A));
        Esa = [Esa eigens0(ix)];                
    end    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
