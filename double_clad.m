um = 1e-6;
Nx = 1000;
n_silica = 1.45;
n_air = 1.0;
r_core = 25.5 * um;
r_clad = 34.0 * um;
r_total = r_core + r_clad;
x = linspace(-25.5 * um, 25.5 * um, Nx);
y = x;
[x_mesh, y_mesh] = meshgrid(x, y);
r_mesh = sqrt(x_mesh.^2 + y_mesh.^2);
n = ones(Nx, Nx) * n_silica;
n(r_mesh < r_total) = n_silica;
n(r_mesh < (r_total - r_clad)) = 1;
n(r_mesh > r_total) = 1;

% Ellipses for glass
glass_ellipses = [
    struct('center', [-19.35 * um, 4.72 * um], 'major_axis', 11.28 * um, 'minor_axis', 10.28 * um, 'angle', 166.29),
    struct('center', [-10.30 * um, 17.82 * um], 'major_axis', 9.85 * um, 'minor_axis', 9.82 * um, 'angle', 120.03),
    struct('center', [4.80 * um, 19.92 * um], 'major_axis', 10.42 * um, 'minor_axis', 9.49 * um, 'angle', 76.45),
    struct('center', [17.46 * um, 10.08 * um], 'major_axis', 10.70 * um, 'minor_axis', 9.91 * um, 'angle', 30.00),
    struct('center', [19.58 * um, -4.59 * um], 'major_axis', 10.80 * um, 'minor_axis', 10.09 * um, 'angle', -13.19),
    struct('center', [10.15 * um, -17.15 * um], 'major_axis', 11.20 * um, 'minor_axis', 10.09 * um, 'angle', -59.38),
    struct('center', [-4.73 * um, -19.40 * um], 'major_axis', 11.36 * um, 'minor_axis', 9.97 * um, 'angle', -103.70),
    struct('center', [-16.85 * um, -11.15 * um], 'major_axis', 10.70 * um, 'minor_axis', 10.60 * um, 'angle', -147.51)
];

% Ellipses for air
air_ellipses = [
    struct('center', [-19.35 * um, 4.72 * um], 'major_axis', 10.88 * um, 'minor_axis', 9.88 * um, 'angle', 166.29),
    struct('center', [-10.30 * um, 17.82 * um], 'major_axis', 9.45 * um, 'minor_axis', 9.42 * um, 'angle', 120.03),
    struct('center', [4.80 * um, 19.92 * um], 'major_axis', 10.02 * um, 'minor_axis', 9.09 * um, 'angle', 76.45),
    struct('center', [17.46 * um, 10.08 * um], 'major_axis', 10.30 * um, 'minor_axis', 9.51 * um, 'angle', 30.00),
    struct('center', [19.58 * um, -4.59 * um], 'major_axis', 10.40 * um, 'minor_axis', 9.69 * um, 'angle', -13.19),
    struct('center', [10.15 * um, -17.15 * um], 'major_axis', 10.80 * um, 'minor_axis', 9.69 * um, 'angle', -59.38),
    struct('center', [-4.73 * um, -19.40 * um], 'major_axis', 10.96 * um, 'minor_axis', 9.57 * um, 'angle', -103.70),
    struct('center', [-16.85 * um, -11.15 * um], 'major_axis', 10.30 * um, 'minor_axis', 10.20 * um, 'angle', -147.51)
];

for i = 1:numel(glass_ellipses) + numel(air_ellipses)
    if i <= numel(glass_ellipses)
        ellipse_params = glass_ellipses(i);
        is_glass = true;
    else
        ellipse_params = air_ellipses(i - numel(glass_ellipses));
        is_glass = false;
    end
    
    center = ellipse_params.center;
    major_axis = ellipse_params.major_axis / 2;
    minor_axis = ellipse_params.minor_axis / 2;
    angle = deg2rad(ellipse_params.angle);
    x_rotated = (x_mesh - center(1)) * cos(angle) + (y_mesh - center(2)) * sin(angle);
    y_rotated = (y_mesh - center(2)) * cos(angle) - (x_mesh - center(1)) * sin(angle);
    ellipse_mask = ((x_rotated / major_axis).^2 + (y_rotated / minor_axis).^2) < 1;
    n(ellipse_mask) = n_silica * is_glass + n_air * ~is_glass;
end

figure;
imagesc(x*1e6, x*1e6, n);
axis square;
xlabel('\mum');
ylabel('\mum');
hold on;
