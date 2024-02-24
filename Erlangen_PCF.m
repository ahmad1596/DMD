clc; clear variables; close all;

%% Set up problem

lambda = 650e-9;
k0 = 2*pi/lambda;
beta = k0; % Propagation constant will be close to that of free space.
Nx = 450; % Change Nx to match the new fiber profile
NoModes = 10;

um = 1e-6;
n_silica = 1.45;
n_air = 1.0;
r_core = 25.5 * um;
r_clad = 34.0 * um;
r_total = r_core + r_clad;
x = linspace(8 * um, 20 * um, Nx);
[x_mesh, y_mesh] = meshgrid(x, x.');
r_mesh = sqrt(x_mesh.^2 + y_mesh.^2);
n = ones(Nx, Nx) * n_silica;
n(r_mesh < r_total) = n_silica;
n(r_mesh < (r_total - r_clad)) = 1;
n(r_mesh > r_total) = 1;

glass_ellipses = [
    struct('center', [20 * um, 0], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', 0),
    struct('center', [-20 * um, 0], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', 0),
    struct('center', [0, 20 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', pi/2),
    struct('center', [0, -20 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', pi/2),
    struct('center', [14 * um, 14 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', -pi/4),
    struct('center', [-14 * um, -14 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', -pi/4),
    struct('center', [14 * um, -14 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', pi/4),
    struct('center', [-14 * um, 14 * um], 'major_axis', 6 * um, 'minor_axis', 3 * um, 'angle', pi/4)
];

air_ellipses = [
    struct('center', [20 * um, 0], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', 0),
    struct('center', [-20 * um, 0], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', 0),
    struct('center', [0, 20 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', pi/2),
    struct('center', [0, -20 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', pi/2),
    struct('center', [14 * um, 14 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', -pi/4),
    struct('center', [-14 * um, -14 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', -pi/4),
    struct('center', [14 * um, -14 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', pi/4),
    struct('center', [-14 * um, 14 * um], 'major_axis', 5 * um, 'minor_axis', 2 * um, 'angle', pi/4)
];

% Apply ellipses for glass and air
for i = 1:numel(glass_ellipses) + numel(air_ellipses)
    if i <= numel(glass_ellipses)
        ellipse_params = glass_ellipses(i);
        is_glass = true;
    else
        ellipse_params = air_ellipses(i - numel(glass_ellipses));
        is_glass = false;
    end
    
    center = ellipse_params.center;
    major_axis = ellipse_params.major_axis;
    minor_axis = ellipse_params.minor_axis;
    angle = ellipse_params.angle;
    x_rotated = (x_mesh - center(1)) * cos(angle) - (y_mesh - center(2)) * sin(angle);
    y_rotated = (x_mesh - center(1)) * sin(angle) + (y_mesh - center(2)) * cos(angle);
    ellipse_mask = ((x_rotated / major_axis).^2 + (y_rotated / minor_axis).^2) < 1;
    n(ellipse_mask) = n_silica * is_glass + n_air * ~is_glass;
end

dx = x(2) - x(1);

%% Show refractive index profile

figure;
imagesc(x*1e6, x*1e6, n);
axis square;
xlabel('\mum');
ylabel('\mum');
hold on;

%% Call FD solver

tic;
RetVal = ModeSolverFD(dx, n, lambda, beta, NoModes);
toc;

%% Plot modes

for i = 1:NoModes
    figure;
    imagesc(x*1e6, x*1e6, RetVal.Eabs{i});
    title(['\beta = ' num2str(RetVal.beta(i))]);
    axis square;
end

save('FD Solver Result.mat', 'RetVal', '-v7.3');

function [x,n] = GenerateFibreProfile(Nx)

end
