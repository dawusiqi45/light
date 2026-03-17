% 2025-11-11. 优化简化 - Sinc光源版 (并行稳定版 - 防止内存溢出)
% 修复: 自动检测内存占用，限制并行核心数，防止 Worker 崩溃

% =========================================================================
% 第一部分：基本工具函数
% =========================================================================

% 二维傅里叶变换函数
function G = ft2(g, delta)
    G = fftshift(fft2(fftshift(g))) * delta^2;
end

% 二维逆傅里叶变换函数
function g = ift2(G, delta_f)
    N = size(G, 1);
    g = ifftshift(ifft2(ifftshift(G))) * (N * delta_f)^2;
end

function z = circ(x, y, D)
    r = sqrt(x.^2 + y.^2);
    z = double(r < D/2);
    z(r == D/2) = 0.5;
end

function window = create_absorption_window(N, width)
    [x, y] = meshgrid(-N/2:N/2-1, -N/2:N/2-1);
    r = sqrt(x.^2 + y.^2);
    radius = N/2;
    sigma = width * radius;
    window = exp(-(r/sigma).^2 .* (r > radius*0.7));
    window(r < radius*0.7) = 1;
end

% =========================================================================
% 第二部分：相位屏生成
% =========================================================================
function [phz_screen] = ft_sh_phase_screen(r0, N, delta, L0, l0)
    D = N*delta;
    
    del_f = 1/(N*delta);
    fx = (-N/2 : N/2-1) * del_f;
    [fx_h, fy_h] = meshgrid(fx);
    [~, f] = cart2pol(fx_h, fy_h);
    f(f == 0) = eps;
    
    fm = 5.92/l0/(2*pi);
    f0 = 1/L0;
    
    PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ./ (f.^2 + f0^2).^(11/6);
    PSD_phi(N/2+1, N/2+1) = 0;
    
    cn = (randn(N) + 1i*randn(N)) .* sqrt(PSD_phi)*del_f;
    phz_hi = real(ift2(cn, 1));
    
    [x, y] = meshgrid((-N/2 : N/2-1) * delta);
    phz_lo = zeros(size(phz_hi));
    
    for p = 1:3
        del_f = 1 / (3^p*D);
        fx = (-1 : 1) * del_f;
        [fx_sub, fy_sub] = meshgrid(fx);
        [~, f] = cart2pol(fx_sub, fy_sub);
        f(f == 0) = eps;
        
        fm = 5.92/l0/(2*pi);
        f0 = 1/L0;
        
        PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ./ (f.^2 + f0^2).^(11/6);
        PSD_phi(2, 2) = 0;
        
        cn = (randn(3) + 1i*randn(3)) .* sqrt(PSD_phi)*del_f;
        SH = zeros(N);
        
        for ii = 1:3
            for jj = 1:3
                SH = SH + cn(ii,jj) * exp(1i*2*pi*(fx_sub(ii,jj)*x + fy_sub(ii,jj)*y));
            end
        end
        phz_lo = phz_lo + SH;
    end
    
    phz_lo = real(phz_lo) - mean(real(phz_lo(:)));
    phz_screen = phz_lo + phz_hi;
end

% =========================================================================
% 第三部分：传播算法
% =========================================================================

function [U2] = prop_angular_spectrum(U1, wvl, d1, d2, z)
    N = size(U1, 1);
    k = 2*pi/wvl;
    
    absorption_window = create_absorption_window(N, 0.15);
    
    [x1, y1] = meshgrid((-N/2 : N/2-1) * d1);
    r1sq = x1.^2 + y1.^2;
    
    [x2, y2] = meshgrid((-N/2 : N/2-1) * d2);
    r2sq = x2.^2 + y2.^2;
    
    m = d2/d1;
    
    deltaf = 1/(N*d1);
    [fx, fy] = meshgrid((-N/2 : N/2-1) * deltaf);
    fsq = fx.^2 + fy.^2;
    
    U1 = U1 .* absorption_window;
    
    Q1 = exp(1i*k/2*(1-m)/z * r1sq);
    Q2 = exp(-1i*pi^2*2*z/m/k * fsq);
    Q3 = exp(1i*k/2*(m-1)/(m*z) * r2sq);
    
    U2 = Q3 .* ift2(Q2 .* ft2(U1 .* Q1, d1) / m, deltaf);
    
    U2 = U2 .* absorption_window;
end

function [U_out] = prop_through_screens(U_in, wvl, delta1, deltan, z_screens, phz_screens)
    n_screens = length(z_screens);
    dz = diff([0 z_screens]);
    alpha = [0 z_screens] / z_screens(end);
    delta = (1-alpha) * delta1 + alpha * deltan;
    
    U_current = U_in;
    
    for i = 1:n_screens
        U_current = prop_angular_spectrum(U_current, wvl, delta(i), delta(i+1), dz(i));
        U_current = U_current .* exp(1i * phz_screens(:,:,i));
    end
    
    U_out = U_current;
end

% =========================================================================
% 第四部分：参数优化
% =========================================================================
function r0_screens = calculate_screen_r0(num_screens, z_screens, L, Cn2_profile, wvl)
    k = 2*pi/wvl;
 
    if isscalar(Cn2_profile)
        cn2_func = @(z) Cn2_profile;
    else
        cn2_func = Cn2_profile;
    end

    dz = L / 200; 
    z_vec = dz:dz:L;
    r0_term = 0;
    theta0_term = 0;
    sigma2_chi_term = 0;
    for i = 1:length(z_vec)
        z = z_vec(i);
        cn2 = cn2_func(z);

        r0_term = r0_term + cn2 * (z/L)^(5/3) * dz;
        theta0_term = theta0_term + cn2 * (1-z/L)^(5/3) * dz;
        sigma2_chi_term = sigma2_chi_term + cn2 * (z/L)^(5/6) * (1-z/L)^(5/6) * dz;
    end
    r0_total = (0.423 * k^2 * r0_term)^(-3/5);
    theta0 = (2.91 * k^2 * L^(5/3) * theta0_term)^(-3/5);
    sigma2_chi = 0.563 * k^(7/6) * L^(5/6) * sigma2_chi_term;

    n_optimize = num_screens - 1;
    fprintf('初始化参数优化...\n');
    dz_segment = L / num_screens;
    r0_init = zeros(1, n_optimize);
    
    for i = 1:n_optimize
        z_mid = z_screens(i);
        cn2_avg = cn2_func(z_mid);
        if cn2_avg > eps
            r0_init(i) = (0.423 * k^2 * cn2_avg * dz_segment)^(-3/5);
        else
            r0_init(i) = Inf;
        end
    end

    function [c, ceq] = constraints(r0_vec)
        log_amp_var_contributions = zeros(1, length(r0_vec));
        for i = 1:length(r0_vec)
            if r0_vec(i) > 0
                z = z_screens(i);
                log_amp_var_contributions(i) = 1.331 * k^(-5/6) * L^(5/6) * ...
                    r0_vec(i)^(-5/3) * (z/L)^(5/6) * (1-z/L)^(5/6);
            end
        end
        max_ratio = 0.20; 
        c = log_amp_var_contributions / sigma2_chi - max_ratio;
        ceq = [];
    end

    function f = objective(r0_vec)
        b = [r0_total^(-5/3);  
             sigma2_chi/(1.331*k^(-5/6)*L^(5/6)); 
             theta0^(-5/3)/(6.8794*L^(5/3))];

        A = zeros(3, length(r0_vec)); 
        for i = 1:length(r0_vec)
            z = z_screens(i);
            A(1, i) = (z/L)^(5/3);
            A(2, i) = (z/L)^(5/6) * (1-z/L)^(5/6);
            A(3, i) = (1-z/L)^(5/3);
        end

        R = zeros(length(r0_vec), 1);
        for i = 1:length(r0_vec)
            if r0_vec(i) > 0
                R(i) = r0_vec(i)^(-5/3);
            end
        end
        
        b_hat = A * R;
        f = norm(b - b_hat)^2;
    end
    fprintf('正在优化相位屏Fried参数...\n');
    
    options = optimoptions('fmincon', ...
        'Display', 'off', 'Algorithm', 'interior-point', 'MaxIterations', 2000, ...
        'MaxFunctionEvaluations', 10000, 'OptimalityTolerance', 1e-8, ...
        'ConstraintTolerance', 1e-8, 'StepTolerance', 1e-10);
    
    lb = zeros(1, n_optimize);
    ub = Inf * ones(1, n_optimize);

    [r0_opt, fval, exitflag] = fmincon(@objective, r0_init, [], [], [], [], ...
                                       lb, ub, @constraints, options);

    if exitflag <= 0
        warning('优化可能未收敛，exitflag = %d', exitflag);
    end

    r0_screens = [r0_opt, Inf];

    r0_opt_total = 0;
    for i = 1:n_optimize
        z = z_screens(i);
        r0_opt_total = r0_opt_total + r0_opt(i)^(-5/3) * (z/L)^(5/3);
    end
    r0_opt_total = r0_opt_total^(-3/5);

    fprintf('========================================\n');
    fprintf('弗里德参数 r0 (m)    %.5f     %.5f\n', r0_total, r0_opt_total);
    fprintf('========================================\n\n');
end

% =========================================================================
% 第五部分：PSF生成与保存 (核心修复：内存保护机制)
% =========================================================================

function filename = generate_psf_filename(image_size, psf_grid_size, skip, wvl, L, D, focal_length, Cn2, N, num_screens)
    if isa(Cn2, 'function_handle')
        cn2_str_simple = 'func_profile'; 
        param_hash = sprintf('psf_%dx%d_psf%dx%d_skip%d_wvl%.0f_L%d_D%.4f_f%.1f_Cn2_%s_N%d_ns%d_SINC_CORRECTED.mat', ...
            image_size(1), image_size(2), psf_grid_size(1), psf_grid_size(2), skip, ...
            wvl*1e9, L, D, focal_length, cn2_str_simple, N, num_screens);
    else
        param_hash = sprintf('psf_%dx%d_psf%dx%d_skip%d_wvl%.0f_L%d_D%.4f_f%.1f_Cn2%.0e_N%d_ns%d_SINC_CORRECTED.mat', ...
            image_size(1), image_size(2), psf_grid_size(1), psf_grid_size(2), skip, ...
            wvl*1e9, L, D, focal_length, Cn2, N, num_screens);
    end
    filename = param_hash;
end

function save_psf_array(psf_array, filename, params)
    fprintf('正在保存PSF阵列到文件: %s\n', filename);
    save(filename, 'psf_array', 'params', '-v7.3');
    fprintf('PSF阵列保存完成。\n');
end

function [psf_array, params] = load_psf_array(filename)
    fprintf('正在从文件加载PSF阵列: %s\n', filename);
    data = load(filename);
    psf_array = data.psf_array;
    if isfield(data, 'params')
        params = data.params;
    else
        params = [];
    end
    fprintf('PSF阵列加载完成。\n');
end

% 生成PSF核心函数
function [psf_array] = generate_psf_array(object_grid_size, psf_grid_size, skip, wvl, L, D, focal_length, Cn2_profile, phase_screens_params)
    N = phase_screens_params.N;
    l0 = phase_screens_params.l0;
    L0 = phase_screens_params.L0;
    num_screens = phase_screens_params.num_screens;
    
    psf_filename = generate_psf_filename(object_grid_size, psf_grid_size, skip, wvl, L, D, focal_length, Cn2_profile, N, num_screens);
    
    if exist(psf_filename, 'file')
        fprintf('发现已存在的PSF文件，正在加载...\n');
        [psf_array, ~] = load_psf_array(psf_filename);
        return;
    end
    
    % === 初始化 ===
    fprintf('开始生成PSF阵列...\n');
    delta_o = wvl * L / (2 * D); 
    delta_f = wvl * focal_length / (2 * D); 
    
    z_screens_all = linspace(0, L, num_screens + 1);
    z_screens = z_screens_all(2:end-1); 
    
    r0_screens = calculate_screen_r0(num_screens, z_screens, L, Cn2_profile, wvl);
    
    delta_screen = sqrt(wvl * L / N);
    screen_width = N * delta_screen;
    
    max_object_radius = max(object_grid_size) * delta_o / 2;
    max_angle = max_object_radius / L;
    max_z_screen = max(z_screens);
    max_offset = max_angle * max_z_screen;
    safety_margin = screen_width * 0.2; 
    required_radius = screen_width/2 + max_offset + safety_margin;
    
    % 计算扩展屏幕尺寸
    N_extended = ceil(2 * required_radius / delta_screen);
    N_extended = 2^nextpow2(N_extended); 
    extended_screen_width = N_extended * delta_screen;
    
    fprintf('传播相位屏: %d×%d 像素\n', N, N);
    fprintf('扩展相位屏: %d×%d 像素\n', N_extended, N_extended);
    
    % === [内存安全检测] ===
    % 计算 extended_phz_screens (double complex) 的大小
    % 8 bytes * 2 (complex) = 16 bytes per element
    array_size_bytes = N_extended^2 * (num_screens-1) * 16;
    array_size_gb = array_size_bytes / 1024^3;
    fprintf('注意: 相位屏数组大小约为 %.2f GB\n', array_size_gb);
    
    % 根据数据大小智能设置 Worker 数量
    % 如果数据 > 0.5 GB，限制 worker 数量，防止内存溢出
    current_pool = gcp('nocreate');
    if ~isempty(current_pool)
        fprintf('正在清理旧的并行池以释放内存...\n');
        delete(current_pool);
    end
    
    if array_size_gb > 1.5
        target_workers = 1; % 内存太大，不建议并行，或者只开2个
        fprintf('警告: 数据过大 (>1.5GB)，将限制使用单核或双核以防崩溃。\n');
    elseif array_size_gb > 0.5
        target_workers = 2; % 中等大小，限制为2核
        fprintf('数据较大 (>0.5GB)，将限制并行核心数为 2。\n');
    else
        target_workers = 4; % 默认上限
    end
    
    % 启动受限的并行池
    fprintf('正在启动并行池 (Workers: %d)...\n', target_workers);
    try
        parpool(target_workers);
    catch
        fprintf('无法启动指定数量的Worker，尝试默认配置...\n');
        parpool; 
    end

    % === 生成数据 ===
    nx = ceil(object_grid_size(2) / skip); 
    ny = ceil(object_grid_size(1) / skip);
    psf_array = zeros(psf_grid_size(1), psf_grid_size(2), ny, nx);
    
    fprintf('正在生成扩展相位屏...\n');
    extended_phz_screens = zeros(N_extended, N_extended, num_screens-1);
    for i = 1:num_screens-1
        extended_phz_screens(:,:,i) = ft_sh_phase_screen(r0_screens(i), N_extended, delta_screen, L0, l0);
        fprintf('  已生成相位屏 %d/%d\n', i, num_screens-1);
    end
    
    center_idx = N_extended / 2 + 1;
    
    % === Sinc 点光源构造 ===
    fprintf('正在构造Sinc点光源模型...\n');
    D_tilde = D * 1.5; 
    alpha_param = D_tilde / (wvl * L); 
    
    [x_source, y_source] = meshgrid((-N/2 : N/2-1) * delta_screen);
    k_wave = 2 * pi / wvl;
    
    quad_phase = exp(-1i * k_wave / (2 * L) * (x_source.^2 + y_source.^2));
    sinc_term = sinc(alpha_param * x_source) .* sinc(alpha_param * y_source);
    gauss_win = exp(-(alpha_param^2 / 16) * (x_source.^2 + y_source.^2));
    point_source = quad_phase .* sinc_term .* gauss_win;
    point_source = point_source / sqrt(sum(abs(point_source(:)).^2));
    
    % === 预先计算光瞳掩膜 ===
    [x_pupil_grid, y_pupil_grid] = meshgrid((-N/2 : N/2-1) * delta_screen);
    aperture = circ(x_pupil_grid, y_pupil_grid, D);
    r2_pupil = x_pupil_grid.^2 + y_pupil_grid.^2;
    collimation_phase = exp(1i * k_wave / (2 * L) * r2_pupil);
    total_pupil_mask = aperture .* collimation_phase;

    % =========================================================================
    % [使用 parallel.pool.Constant]
    % =========================================================================
    fprintf('正在构建并行数据常量... \n');
    D_extended_phz_screens = parallel.pool.Constant(extended_phz_screens);
    D_point_source = parallel.pool.Constant(point_source);
    D_total_pupil_mask = parallel.pool.Constant(total_pupil_mask);
    D_z_screens = parallel.pool.Constant(z_screens);
    
    screens_z_pos = [z_screens, L]; 
    D_screens_z_pos = parallel.pool.Constant(screens_z_pos);

    % === 并行计算 PSF 阵列 ===
    fprintf('开始并行生成PSF阵列 (parfor)...\n');
    
    parfor ix = 1:nx
        % 获取数据副本
        local_phz_screens = D_extended_phz_screens.Value;
        local_source = D_point_source.Value;
        local_mask = D_total_pupil_mask.Value;
        local_z = D_z_screens.Value;
        local_z_pos = D_screens_z_pos.Value;
        
        psf_col_slice = zeros(psf_grid_size(1), psf_grid_size(2), ny);
        
        for iy = 1:ny
            x_obj = (ix-1) * skip * delta_o - (object_grid_size(2)-1)/2 * delta_o;
            y_obj = (iy-1) * skip * delta_o - (object_grid_size(1)-1)/2 * delta_o;
            theta_x = x_obj / L;
            theta_y = y_obj / L;
            
            phz_screens_cropped = zeros(N, N, num_screens-1);
            for i = 1:num_screens-1
                offset_x = theta_x * local_z(i);
                offset_y = theta_y * local_z(i);
                offset_ix = round(offset_x / delta_screen); 
                offset_iy = round(offset_y / delta_screen);
                
                row_start = center_idx + offset_iy - N/2;
                col_start = center_idx + offset_ix - N/2;
                
                row_start = max(1, min(row_start, N_extended - N + 1));
                col_start = max(1, min(col_start, N_extended - N + 1));

                phz_screens_cropped(:,:,i) = local_phz_screens(...
                    row_start:row_start+N-1, col_start:col_start+N-1, i);
            end
            
            U_pupil = prop_through_screens(local_source, wvl, delta_screen, delta_screen, ...
                local_z_pos, cat(3, phz_screens_cropped, zeros(N,N)));
            
            U_pupil = U_pupil .* local_mask;
            
            psf_field = ft2(U_pupil, delta_screen);
            psf = abs(psf_field).^2;
            
            delta_focal_plane = focal_length * wvl / (N * delta_screen);
            resize_factor = delta_focal_plane / delta_f;
            psf_resampled = imresize(psf, resize_factor, 'bilinear');
            
            [h_res, w_res] = size(psf_resampled);
            psf_final = zeros(psf_grid_size);
            h_target = psf_grid_size(1);
            w_target = psf_grid_size(2);
            
            h_start_src = max(1, round(h_res/2 - h_target/2) + 1);
            h_len = min(h_target, h_res - h_start_src + 1);
            h_start_tgt = max(1, round(h_target/2 - h_len/2) + 1);
            
            w_start_src = max(1, round(w_res/2 - w_target/2) + 1);
            w_len = min(w_target, w_res - w_start_src + 1);
            w_start_tgt = max(1, round(w_target/2 - w_len/2) + 1);

            if h_len > 0 && w_len > 0
                psf_final(h_start_tgt:h_start_tgt+h_len-1, w_start_tgt:w_start_tgt+w_len-1) = ...
                    psf_resampled(h_start_src:h_start_src+h_len-1, w_start_src:w_start_src+w_len-1);
            end
            
            psf_sum = sum(psf_final(:));
            if psf_sum > eps
                psf_final = psf_final / psf_sum;
            else
                psf_final = psf_final / (eps + psf_sum);
            end
            
            psf_col_slice(:,:,iy) = psf_final;
        end
        
        psf_array(:,:,:,ix) = psf_col_slice;
        % parfor内使用fprintf可能会乱序，但能表示在运行
        fprintf('Worker完成一列计算 (ix=%d)\n', ix);
    end
    
    params = struct();
    params.object_grid_size = object_grid_size;
    params.psf_grid_size = psf_grid_size;
    params.skip = skip;
    params.wvl = wvl;
    params.L = L;
    params.D = D;
    params.focal_length = focal_length;
    params.Cn2_profile = Cn2_profile;
    params.phase_screens_params = phase_screens_params;
    params.N_extended = N_extended;
    params.generation_time = datestr(now);
    
    save_psf_array(psf_array, psf_filename, params);
end

%psf退化成像
function [degraded_image] = apply_psf_degradation(ideal_image, psf_array, skip)
    [M, N] = size(ideal_image);
    [psf_h, psf_w, ny, nx] = size(psf_array);
    
    degraded_image = zeros(M, N);
    
    fprintf('正在插值PSF阵列...\n');
    if skip > 1
        [Xi, Yi] = meshgrid(1:N, 1:M); 
        interpolated_psfs = zeros(psf_h, psf_w, M, N);
        [X_orig_nd, Y_orig_nd] = ndgrid((0:ny-1)*skip + 1, (0:nx-1)*skip + 1);

        parfor h = 1:psf_h
            slice_interpolated = zeros(psf_w, M, N); 
            for w = 1:psf_w
                psf_slice = squeeze(psf_array(h, w, :, :)); % (ny, nx)
                F = griddedInterpolant(X_orig_nd, Y_orig_nd, psf_slice, 'linear', 'none');
                interpolated_slice = F(Yi, Xi);
                interpolated_slice(isnan(interpolated_slice)) = 0; 
                slice_interpolated(w, :, :) = interpolated_slice;
            end
            interpolated_psfs(h, :, :, :) = slice_interpolated;
            if mod(h, 10) == 0
               fprintf('  已插值PSF深度: %d/%d\n', h, psf_h);
            end
        end
    else
        interpolated_psfs = psf_array;
    end
    fprintf('PSF插值完成。\n');

    % 卷积
    fprintf('正在应用空间变化的卷积...\n');
    pad_size = floor(psf_h / 2);
    padded_image = padarray(ideal_image, [pad_size pad_size], 'replicate');
    
    parfor y = 1:M
        row_result = zeros(1, N);
        for x = 1:N
            current_psf = squeeze(interpolated_psfs(:, :, y, x));
            image_patch = padded_image(y : y + psf_h - 1, x : x + psf_w - 1);
            flipped_psf = rot90(current_psf, 2);
            row_result(x) = sum(image_patch .* flipped_psf, 'all');
        end
        degraded_image(y, :) = row_result;
        if mod(y, 50) == 0
            fprintf('  已处理图像: %d/%d 行\n', y, M);
        end
    end
    fprintf('卷积完成。\n');
end

%% 主程序

function anisoplanatic_simulation()
    % 光学参数
    D = 0.2034;
    focal_length = 1.2;
    wvl = 525e-9;
    L = 1000; 
    
    % 湍流参数
    Cn2 = 7e-16;
    l0 = 0.01;
    L0 = 300;
    
    % 相位屏参数
    phase_screens_params = struct();
    phase_screens_params.N = 512;
    phase_screens_params.num_screens = 10;
    phase_screens_params.l0 = l0;
    phase_screens_params.L0 = L0;
    
    % 加载图像
    image_path = 'E:\t_pic\p1\left\left_frame_0000_img_0001_framenum_1.tif';
    
    try
        test_image = imread(image_path);
        fprintf('成功加载图像: %s\n', image_path);
        image_size = size(test_image);
        fprintf('原始图像尺寸: %d x %d\n', image_size(1), image_size(2));
        
        max_size = 600;
        if max(image_size) > max_size
            scale_factor = max_size / max(image_size);
            new_size = round(image_size * scale_factor);
            test_image = imresize(test_image, new_size);
            image_size = size(test_image);
            fprintf('调整后图像尺寸: %d x %d\n', image_size(1), image_size(2));
        end
        
        if isa(test_image, 'uint8')
            test_image = double(test_image) / 255;
        elseif isa(test_image, 'uint16')
            test_image = double(test_image) / 65535;
        else
            test_image = double(test_image);
            test_image = (test_image - min(test_image(:))) / (max(test_image(:)) - min(test_image(:)));
        end
        
    catch ME
        fprintf('加载图像时出错: %s\n', ME.message);
        test_image = checkerboard(16, 16, 16);
        image_size = size(test_image);
    end
    
    % 计算PSF尺寸
    theoretical_psf_width = 2.44 * wvl * focal_length / D;
    pixel_size = wvl * focal_length /(2* D); 
    psf_width_pixels = ceil(5 * theoretical_psf_width / pixel_size);
    
    psf_width_pixels = max(15, psf_width_pixels);
    psf_width_pixels = psf_width_pixels + mod(psf_width_pixels + 1, 2); 
    psf_size = [psf_width_pixels, psf_width_pixels];
    
    fprintf('自动计算的PSF尺寸: %d x %d\n', psf_size(1), psf_size(2));
    
    % 计算skip参数
    total_pixels = image_size(1) * image_size(2);
    if total_pixels > 65536
        skip = 8;
    elseif total_pixels > 16384
        skip = 4;
    elseif total_pixels > 4096
        skip = 4;
    else
        skip = 2;
    end
    
    fprintf('自动计算的skip参数: %d\n', skip);
    
    % 生成PSF阵列（使用扩展相位屏 + 并行计算）
    psf_array = generate_psf_array(image_size, psf_size, skip, wvl, L, D, focal_length, Cn2, phase_screens_params);
    
    % 显示PSF阵列示例
    figure;
    subplot(2,2,1); imagesc(psf_array(:,:,1,1)); title('PSF(1,1)'); axis equal tight; colorbar;
    subplot(2,2,2); imagesc(psf_array(:,:,end,1)); title('PSF(end,1)'); axis equal tight; colorbar;
    subplot(2,2,3); imagesc(psf_array(:,:,1,end)); title('PSF(1,end)'); axis equal tight; colorbar;
    subplot(2,2,4); imagesc(psf_array(:,:,end,end)); title('PSF(end,end)'); axis equal tight; colorbar;
    
    % 应用PSF退化
    fprintf('正在应用非等晕退化...\n');
    degraded_image = apply_psf_degradation(test_image, psf_array, skip);
    
    % 显示原始和退化图像
    figure;
    subplot(1,2,1); imshow(test_image); title('原始图像');
    subplot(1,2,2); imshow(degraded_image, []); title('非等晕退化图像');
    
    fprintf('模拟完成。\n');
    
    save('anisoplanatic_simulation_results.mat', 'psf_array', 'test_image', 'degraded_image');
    imwrite(uint8(test_image * 255), 'original_image.png');
    degraded_normalized = (degraded_image - min(degraded_image(:))) / (max(degraded_image(:)) - min(degraded_image(:)));
    imwrite(uint8(degraded_normalized * 255), 'degraded_image.png');
    
    fprintf('结果已保存。\n');
end

% 执行主程序
anisoplanatic_simulation();