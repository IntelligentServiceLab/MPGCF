ds_list = {
    [32, 64, 128, 256, 512], 
    [0.4, 0.5, 0.6, 0.7, 0.8], 
    [0.25, 0.5, 0.75, 1.00, 1.25], 
    [0.75, 1.00, 1.25, 1.5, 1.75], 
    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],  
    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
};

metrics = { 
    % Recall ndcg coverage tail
    [0.39568, 0.46181 , 0.47269 , 0.49908 ,0.48838 ;
     0.30111 , 0.34793 , 0.35772 , 0.40283 ,0.39338 ;
     0.55858 , 0.65481 , 0.67155 , 0.69665 , 0.63113 ;
     0.14601 , 0.15145 , 0.17951 , 0.19901 ,0.15903 ],

 [0.45663 , 0.46271 , 0.48141 , 0.49908 , 0.51603 ;
     0.37463 , 0.38545 , 0.39001 , 0.40283 , 0.41881 ;
     0.73640 , 0.71293 , 0.69944 , 0.69665 , 0.62435 ;
     0.23537 , 0.21557 , 0.20214 , 0.19901 , 0.17580 ],

    [0.47496 , 0.49908  , 0.47917 , 0.33029 , 0.30755 ;
     0.35309 , 0.40283 , 0.37418 , 0.23145 , 0.21635 ;
     0.63180 , 0.69665 , 0.69865 , 0.75314 , 0.77197 ;
     0.14005 , 0.19901 , 0.20945 , 0.37426 , 0.45485 ],

     [0.51209 ,0.48598 , 0.49908 ,0.42569 , 0.37321 ;
     0.40345 , 0.40277 , 0.40283 , 0.34605 , 0.32037 ;
     0.63632 , 0.66075 , 0.69665 , 0.74778 , 0.78314 ;
     0.16400 , 0.16771 , 0.19901 , 0.33207 , 0.46756],
     
    [0.47743 , 0.48870  , 0.49341 , 0.49908 , 0.49077 ;
     0.36317  , 0.38148 , 0.39941 ,0.40283 , 0.39031 ;
     0.62657 , 0.67364 , 0.68411 , 0.69665 , 0.69047 ;
     0.16161 , 0.16949 , 0.18417 , 0.19901 , 0.18076 ],
     
    [0.49167 , 0.49908 , 0.48637 , 0.47630 , 0.47233 ;
     0.38956 , 0.40283 , 0.38365 , 0.37411 , 0.37037 ;
    0.65312, 0.69665 ,0.64225, 0.59728 ,0.56694 ;
     0.18751 ,0.19901 ,0.18819 ,0.16538 ,0.16384 
],

};

xlabels = {
    '\( d_l \)', 
    '\(\alpha\)', 
    '\( r_{acc} \)', 
    '\( r_{n-acc} \)', 
     '\( \lambda_1 \)',
    '\( \lambda_2 \)', 

};

titles = {
    '(a) The effect of $d_l$', 
    '(b) The effect of  \(\alpha\)', 
    '(c) The effect of $r_{acc}$', 
    '(d) The effect of $r_{n-acc}$', 
    '(e) The effect of \( \lambda_1 \)',
    '(f) The effect of \( \lambda_2 \)' 

};


% 设定画布大小 (30x40 cm)，减少白边
figure('Units', 'centimeters', 'Position', [5, 5, 23, 35]);

% 设定子图布局参数
rows = 3;
cols = 2;

left_margin = 0.04;  % 左边距
bottom_margin = 0.03; % 底边距
plot_width = 0.43;    % 每个子图的宽度
plot_height = 0.24;   % 每个子图的高度
x_spacing = 0.08;    % 水平间距
y_spacing = 0.072;    % 垂直间距

for i = 1:6
    row = floor((i-1) / cols);  % 计算当前子图所在的行
    col = mod(i-1, cols);       % 计算当前子图所在的列

    % 计算子图的 Position（[left, bottom, width, height]）  
    left = left_margin + col * (plot_width + x_spacing);
    bottom = 1 - (row + 1) * (plot_height + y_spacing);

    % 创建子图，并设置 Position
    ax = subplot(rows, cols, i);
    set(ax, 'Position', [left, bottom, plot_width, plot_height]);

    % 数据处理
    recall = metrics{i}(1, :);
    ndcg = metrics{i}(2, :);
    coverage = metrics{i}(3, :);
    tail = metrics{i}(4, :);
    x_new = 1:length(ds_list{i});
    
    % 画图
    plot(x_new, coverage, 'r-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Coverage');hold on;
    plot(x_new, recall, 'g-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Recall'); 
    plot(x_new, ndcg, 'y-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Ndcg','Color', [0.5, 0, 0.5]);
    plot(x_new, tail, 'b-x', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Tail');

    ylim([0, 1]);
    set(gca, 'XTick', x_new);

    % 处理 X 轴标签
    if i == 5 || i == 6
        set(gca, 'XTickLabel', {'$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'}, 'TickLabelInterpreter', 'latex');
    else
        set(gca, 'XTickLabel', ds_list{i});
    end

    % 添加 xlabel 作为子图标题
    xlabel(titles{i}, 'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
    
    % 调整 xlabel 位置，向下移动
    currentPos = get(get(ax, 'xlabel'), 'Position');
    set(get(ax, 'xlabel'), 'Position', [currentPos(1), currentPos(2) - 0.09, currentPos(3)]);

    % 添加 xlabels 到子图上方
    text('Units', 'normalized', 'Position', [0.5, -0.14], 'String', xlabels{i}, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');
    
    legend('show', 'Location', 'northwest');
    grid on;

    % 调整 x 轴范围，避免数据点贴边
    padding = 0.1;
    x_range = max(x_new) - min(x_new);
    xlim([min(x_new) - padding * x_range, max(x_new) + padding * x_range]);
end

% 导出图片
print(gcf, 'output.png', '-dpng', '-r600');
