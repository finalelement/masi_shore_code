function ipmi_2019_represent_error_figure


    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\mse_zeta_optimized.mat')
    figure(1)
    
    %mse_lambda = 
    %b3k_mse(b3k_mse
    bins = linspace(0.00001,0.0005,100);
    bin_index = [1 10 20 30 40 50 60 70 80 90 100];
    x_labels = bins(bin_index);
    
    [h1,p1] = hist(b3k_mse,bins);
    [h2,p2] = hist(b6k_mse,bins);
    [h3,p3] = hist(b9k_mse,bins);
    [h4,p4] = hist(b12k_mse,bins);
    [h5,p5] = hist(b3k_b6k_mse,bins);
    [h6,p6] = hist(b3k_b9k_mse,bins);
    
    plot(h1,'r--','LineWidth',1.5)
    hold on
    plot(h2,'g-.','LineWidth',1.5)
    plot(h3,'b-.','LineWidth',1.5)
    plot(h4,'c-.','LineWidth',1.5)
    %plot(h5,'m-.','LineWidth',1.5)
    %plot(h6,'k-.','LineWidth',1.5)
    
    set(gca, 'YScale', 'log')
    set(gca, 'XTickLabels', x_labels)
end