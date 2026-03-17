% 多相位屏输出
% 点光源（Sinc模型）通过大气湍流传播仿真
% 包含透镜聚焦成像过程 (严格遵循 Hardie et al. 2017 Eq.18, 22, 26)
% 修改：增加入瞳前完整光场的可视化

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
% 第二部分：相位屏生成 (保留亚谐波)
% =========================================================================
function [phz_screen] = ft_sh_phase_screen(r0, N, delta, L0, l0)
    D = N*delta;
    
    % 高频部分(FFT方法)
    del_f = 1/(N*delta);
    fx = (-N/2 : N/2-1) * del_f;
    [fx_h, fy_h] = meshgrid(fx);
    [~, f] = cart2pol(fx_h, fy_h);
    f(f == 0) = eps;
    
    fm = 5.92/l0/(2*pi);
    f0 = 1/L0;
    
    % 修正的冯·卡门大气相位PSD
    PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ./ (f.^2 + f0^2).^(11/6);
    PSD_phi(N/2+1, N/2+1) = 0;
    
    cn = (randn(N) + 1i*randn(N)) .* sqrt(PSD_phi)*del_f;
    
    phz_hi = real(ift2(cn, 1));
    
    [x, y] = meshgrid((-N/2 : N/2-1) * delta);
    
    phz_lo = zeros(size(phz_hi));
    
    % 亚谐波补偿循环
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

% 角谱衍射
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

% 多相位屏波传播函数
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
% 第四部分：参数优化 (Fried参数分配)
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

    fprintf('\n========================================\n');
    fprintf('弗里德参数 r0       = %.5f m\n', r0_total);
    fprintf('等晕角 θ0           = %.3f μrad\n', theta0*1e6);
    fprintf('对数振幅方差 σ²χ   = %.6f\n', sigma2_chi);
    fprintf('========================================\n\n');

    % 优化前num_screens-1个相位屏
    n_optimize = num_screens - 1;
    
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
    
    % 约束:单个相位屏最大湍流贡献20%约束
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

    % 目标函数
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
        warning('优化可能未收敛,exitflag = %d', exitflag);
    end

    if any(r0_opt <= 0)
        warning('发现非正的r0值,可能需要调整约束或初值');
    end

    r0_screens = [r0_opt, Inf];

    % 计算优化后的结果并输出
    r0_opt_total = 0;
    for i = 1:n_optimize
        z = z_screens(i);
        r0_opt_total = r0_opt_total + r0_opt(i)^(-5/3) * (z/L)^(5/3);
    end
    r0_opt_total = r0_opt_total^(-3/5);

    theta0_opt = 0;
    for i = 1:n_optimize
        z = z_screens(i);
        theta0_opt = theta0_opt + r0_opt(i)^(-5/3) * (1-z/L)^(5/3);
    end
    theta0_opt = (6.8794 * L^(5/3) * theta0_opt)^(-3/5);

    sigma2_chi_opt = 0;
    for i = 1:n_optimize
        z = z_screens(i);
        sigma2_chi_opt = sigma2_chi_opt + r0_opt(i)^(-5/3) * ...
                         (z/L)^(5/6) * (1-z/L)^(5/6);
    end
    sigma2_chi_opt = 1.331 * k^(-5/6) * L^(5/6) * sigma2_chi_opt;

    fprintf('\n========================================\n');
    fprintf('      优化结果对比(理论值 vs 优化值)  \n');
    fprintf('========================================\n');
    fprintf('参数                  理论值        优化值        误差(%%)\n');
    fprintf('----------------------------------------\n');
    fprintf('弗里德参数 r0 (m)    %.5f     %.5f     %+.2f%%\n', ...
        r0_total, r0_opt_total, (r0_opt_total-r0_total)/r0_total*100);
    fprintf('等晕角 θ0 (μrad)     %.3f      %.3f      %+.2f%%\n', ...
        theta0*1e6, theta0_opt*1e6, (theta0_opt-theta0)/theta0*100);
    fprintf('对数振幅方差 σ²χ    %.6f    %.6f    %+.2f%%\n', ...
        sigma2_chi, sigma2_chi_opt, (sigma2_chi_opt-sigma2_chi)/sigma2_chi*100);
    fprintf('========================================\n\n');
end


% =========================================================================
% 第五部分：高精度点光源模型 (Sinc)
% =========================================================================

% Sinc点光源 (基于Hardie et al. 2017 Eq.18)
% 目的：在远场产生均匀振幅分布，模拟理想点源(球面波)
function [U_source] = generate_sinc_point_source(N, delta, wvl, L, D_tilde)
    [x, y] = meshgrid((-N/2 : N/2-1) * delta);
    k = 2 * pi / wvl;

    % alpha 参数决定了Sinc的宽度，从而决定远场照明区域D_tilde
    alpha = D_tilde / (wvl * L);

    % 1. 二次相位因子 (模拟球面波发散)
    quad_phase = exp(-1i * k / (2 * L) * (x.^2 + y.^2));

    % 2. Sinc 函数项 (MATLAB sinc = sin(pi*x)/(pi*x))
    % 在频域(远场)对应矩形函数，即均匀光强
    sinc_term = sinc(alpha * x) .* sinc(alpha * y);

    % 3. 高斯窗函数 (变迹，用于抑制数值传播的振铃效应)
    gauss_win = exp(-(alpha^2 / 16) * (x.^2 + y.^2));

    % 组合生成复振幅场
    U_source = quad_phase .* sinc_term .* gauss_win;

    % 能量归一化
    U_source = U_source / sqrt(sum(abs(U_source(:)).^2));
end

% =========================================================================
% 第六部分：可视化函数 (包含三种光场图)
% =========================================================================

% 绘制光强分布：3D彩色图 + 2D黑白图
function plot_intensity_distribution(U_field, delta, title_str, fig_num)
    N = size(U_field, 1);
    intensity = abs(U_field).^2;
    % 创建坐标轴
    x = ((-N/2 : N/2-1) * delta); 
    y = ((-N/2 : N/2-1) * delta);
    
    % 如果尺度太小，自动转为mm或um
    unit_str = 'm';
    if max(abs(x)) < 1e-3
        x = x * 1e6; y = y * 1e6; unit_str = 'μm';
    elseif max(abs(x)) < 1
        x = x * 1e3; y = y * 1e3; unit_str = 'mm';
    end
    
    [X, Y] = meshgrid(x, y);
    
    figure(fig_num);
    set(gcf, 'Position', [100, 100, 1400, 600]);
    % 3D彩色图
    subplot(1,2,1);
    surf(X, Y, intensity, 'EdgeColor', 'none');
    xlabel(['X (' unit_str ')'], 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(['Y (' unit_str ')'], 'FontSize', 12, 'FontWeight', 'bold');
    zlabel('光强 (归一化)', 'FontSize', 12, 'FontWeight', 'bold');
    title([title_str ' - 三维光强'], 'FontSize', 14, 'FontWeight', 'bold');
    colormap(gca, 'jet');  
    colorbar;
    view(0, 90); % 俯视图
    shading interp;
    axis square;
    grid on;
    set(gca, 'FontSize', 11);
    
    % 2D黑白图
    subplot(1,2,2);
    imagesc(x, y, intensity);
    xlabel(['X (' unit_str ')'], 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(['Y (' unit_str ')'], 'FontSize', 12, 'FontWeight', 'bold');
    title([title_str ' - 二维分布'], 'FontSize', 14, 'FontWeight', 'bold');
    colormap(gca, 'gray');  
    colorbar;
    axis equal tight;
    set(gca, 'FontSize', 11);
end

%3D彩色图 + 2D彩色图 (用于相位屏)
function plot_single_phase_screen(phase_screen, delta, screen_idx, z_position, fig_num)
    N = size(phase_screen, 1);
    % 创建坐标轴(单位:m)
    x = ((-N/2 : N/2-1) * delta);
    y = ((-N/2 : N/2-1) * delta);
    [X, Y] = meshgrid(x, y);

    figure(fig_num);
    set(gcf, 'Position', [100, 100, 1400, 600]);
    
    subplot(1,2,1);
    surf(X, Y, phase_screen, 'EdgeColor', 'none');
    title(sprintf('第%d个相位屏 (z=%.0fm)', screen_idx, z_position));
    colormap(gca, 'jet'); 
    colorbar;
    view(45, 30);
    shading interp;
    
    subplot(1,2,2);
    imagesc(x, y, phase_screen);
    title(sprintf('二维分布 (z=%.0fm)', z_position));
    colorbar;
    axis equal tight;
end

% =========================================================================
% 第七部分:主程序
% =========================================================================

function point_source_turbulence_simulation()
    fprintf('========================================\n');
   
    % 设置光学参数
    D = 0.2034;          % 接收孔径直径 (m)
    wvl = 550e-9;        % 波长 (m)
    L = 1000;            % 传播距离 (m)
    focal_length = 1.2;  % 透镜焦距 (m) - 用于最后的成像聚焦
    
    % [点源参数]
    % D_tilde: 目标照明区域宽度，通常设为孔径D的1.5倍以保证均匀覆盖
    D_tilde = D * 1.5;   
    
    % 湍流参数
    Cn2 =6.44e-16;         % 折射率结构常数
    l0 = 0.01;           % 内尺度 (m)
    L0 = 300;            % 外尺度 (m)
    
    % 相位屏参数
    N = 512;             % 网格点数
    num_screens = 10;    % 相位屏数量
    
    fprintf('仿真参数设置:\n');
    fprintf('  类型: 点光源(Sinc模型)大气传输 -> 透镜成像\n');
    fprintf('  传播距离 L = %.0f m\n', L);
    fprintf('  波长 λ = %.0f nm\n', wvl*1e9);
    fprintf('  接收口径 D = %.4f m\n', D);
    fprintf('  焦距 f = %.2f m\n', focal_length);
    fprintf('  照明覆盖区 D_tilde = %.4f m\n', D_tilde);
    fprintf('  Cn² = %.2e m^(-2/3)\n', Cn2);
    fprintf('  相位屏数量 = %d\n\n', num_screens);
    
    % 计算采样参数
    delta_screen = sqrt(wvl * L / N);
    fprintf('相位屏采样间隔 = %.4f m\n', delta_screen);
    fprintf('相位屏物理尺寸 = %.2f m × %.2f m\n\n', N*delta_screen, N*delta_screen);
    

    z_screens = linspace(0, L, num_screens + 1);
    z_screens = z_screens(2:end-1);
    % [修正] 传播终点位置需包含 z = L，用于补齐最后一段自由传播（末端不加湍流相位）
    z_prop = linspace(0, L, num_screens + 1);
    z_prop = z_prop(2:end);

    
    r0_screens = calculate_screen_r0(num_screens, z_screens, L, Cn2, wvl);
    
    % 生成相位屏
    fprintf('正在生成湍流相位屏...\n');
    phz_screens = zeros(N, N, num_screens);
    for i = 1:num_screens-1
        phz_screens(:,:,i) = ft_sh_phase_screen(r0_screens(i), N, delta_screen, L0, l0);
    end
    % 末端补齐传播平面（z=L）不施加湍流相位
    phz_screens(:,:,num_screens) = 0;
    fprintf('相位屏生成完成!\n\n');
    
    % 生成初始点光源 (Sinc模型)
    fprintf('正在生成Sinc点光源...\n');
    U_initial = generate_sinc_point_source(N, delta_screen, wvl, L, D_tilde);
    fprintf('点光源生成完成!\n\n');
    
    % 通过湍流传播
    fprintf('正在进行湍流传播计算 (Source -> Pupil)...\n');
    U_output = prop_through_screens(U_initial, wvl, delta_screen, delta_screen, z_prop, phz_screens);
    fprintf('传播计算完成!\n\n');
    
    % ---------------------------------------------------------------------
    % [核心修正] 严格按照论文 Eq.22 和 Eq.26 进行成像
    % ---------------------------------------------------------------------
    
    % 1. 生成接收孔径 (Pupil Mask)
    [x_pupil, y_pupil] = meshgrid((-N/2 : N/2-1) * delta_screen);
    r2_pupil = x_pupil.^2 + y_pupil.^2;
    aperture = circ(x_pupil/D, y_pupil/D, 1);
    
    % 2. 计算准直相位补偿 (Collimation Phase - Paper Eq.22)
    k = 2 * pi / wvl;
    collimation_phase = exp(1i * k / (2 * L) * r2_pupil);
    
    % 3. 组合光瞳函数 p(x,y)
    U_pupil_corrected = U_output .* aperture .* collimation_phase;
    
    % 4. FFT计算PSF (Paper Eq.26)
    fprintf('正在进行透镜聚焦 (FFT)...\n');
    U_focal = fftshift(fft2(fftshift(U_pupil_corrected)));
    
    % 计算焦平面采样间隔
    delta_focal = (wvl * focal_length) / (N * delta_screen);
    fprintf('焦平面采样间隔 = %.2f μm\n', delta_focal*1e6);
 
    fprintf('正在生成可视化图像...\n\n');
    

    fprintf('  生成图1: 初始点源分布(近场)...\n');
    plot_intensity_distribution(U_initial, delta_screen, '图1: 初始Sinc点光源', 1);
    
    % [新增] 可视化：入瞳前的光场 (未加孔径限制)
    % 这展示了光到达望远镜位置时的完整分布，说明光其实充满了整个空间
    fprintf('  生成图2: 到达接收面的完整光场(未加光圈)...\n');
    plot_intensity_distribution(U_output, delta_screen, '图2: 到达接收面的完整光场 (未加光圈)', 2);
  
    fprintf('  生成图3: 望远镜入瞳处光场(受孔径限制)...\n');
    plot_intensity_distribution(U_pupil_corrected, delta_screen, '图3: 入瞳处光场 (已加光圈)', 3);
    
    fprintf('  生成图4: 焦平面成像光斑(PSF)...\n');
    plot_intensity_distribution(U_focal, delta_focal, '图4: 焦平面成像 (Focal Plane)', 4);

    % 显示相位屏
    screens_to_plot = [1, 5, 9];
    fig_start_num = 5; 
    for i = 1:length(screens_to_plot)
        screen_index = screens_to_plot(i); 
        if screen_index <= size(phz_screens, 3)
            current_fig_num = fig_start_num + i - 1;
            plot_single_phase_screen(phz_screens(:,:,screen_index), delta_screen, ...
                                    screen_index, z_screens(screen_index), current_fig_num);
        end
    end
    
    fprintf('\n可视化图像生成完成!\n\n');
    
    % 计算并显示统计信息
    fprintf('========================================\n');
    fprintf('           仿真结果统计\n');
    fprintf('========================================\n');
    
    % 接收面光束特性 (PSF)
    intensity_focal = abs(U_focal).^2;
    peak_focal = max(intensity_focal(:));
    
    fprintf('焦平面成像 (PSF):\n');
    fprintf('  峰值光强 = %.4e\n', peak_focal);

    % 计算光斑尺寸 (二阶矩)
    [Yf, Xf] = meshgrid((-N/2 : N/2-1) * delta_focal);
    I_norm = intensity_focal / sum(intensity_focal(:));
    
    cx = sum(Xf(:) .* I_norm(:));
    cy = sum(Yf(:) .* I_norm(:));
    sx = sqrt(sum((Xf(:) - cx).^2 .* I_norm(:)));
    sy = sqrt(sum((Yf(:) - cy).^2 .* I_norm(:)));
    spot_diameter = 2 * sqrt(sx^2 + sy^2);
    
    fprintf('  光斑直径 (1/e²) ≈ %.2f μm\n', spot_diameter*1e6);
    fprintf('  质心偏移 (Tilt) = (%.2f, %.2f) μm\n', cx*1e6, cy*1e6);
    
    fprintf('========================================\n\n');
end

% 执行主程序
point_source_turbulence_simulation();