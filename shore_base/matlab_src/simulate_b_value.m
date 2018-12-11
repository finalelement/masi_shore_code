function simulate_b_value

    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_data.mat')
    b3k = ms_data(:,1:100);
    b6k = ms_data(:,101:200);
    b9k = ms_data(:,201:300);
    b12k = ms_data(:,301:400);
    
    adc=1/log(b3k)/6000;




end