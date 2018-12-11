function create_datasets_diff_combos_of_b_values_ipmi_exp

    % IPMI 2019
    % Train We plan to train on the following listed Combinations
    
    % b3000
    % b9000
    % b12000
    % b3000,b9000
    % b3000,b12000
    % b9000,b12000
    % b3000,b9000,b12000
    
    % We will test on the following combinations
    % b6000
    % b3000,b6000
    % b6000,b9000
    % b6000,b12000
    % b3000,b6000,b9000
    % b3000,b6000,b9000,b12000

    % Define Paths
    data_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\multi_shell_data_b0.mat';
    bvec_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec';
    bval_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval';
    
    % Load Everything
    data = load(data_path);
    data = data.ms_data;
    bvecs = dlmread(bvec_path);
    bvals = dlmread(bval_path);
    
    % Get the b0's
    b0_data = data(:,401);
    b0_bvecs = bvecs(:,401);
    b0_bvals = bvals(401);
    
    % Single Shell Sets (Note, these all include b0's)
    % B3000
    b3k_data(:,1) = b0_data;
    b3k_bvecs(:,1) = b0_bvecs;
    b3k_bvals(1) = b0_bvals;
    
    b3k_data(:,2:101) = data(:,1:100);
    b3k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_bvals(2:101) = bvals(1:100);
    
    save('b3k_data.mat','b3k_data')
    dlmwrite('b3k_bvecs.bvec',b3k_bvecs)
    dlmwrite('b3k_bvals.bval',b3k_bvals)
    % B6000
    b6k_data(:,1) = b0_data;
    b6k_bvecs(:,1) = b0_bvecs;
    b6k_bvals(1) = b0_bvals;
    
    b6k_data(:,2:101) = data(:,101:200);
    b6k_bvecs(:,2:101) = bvecs(:,101:200);
    b6k_bvals(2:101) = bvals(101:200);
    
    save('b6k_data.mat','b6k_data')
    dlmwrite('b6k_bvecs.bvec',b6k_bvecs)
    dlmwrite('b6k_bvals.bval',b6k_bvals)
    % B9000
    b9k_data(:,1) = b0_data;
    b9k_bvecs(:,1) = b0_bvecs;
    b9k_bvals(1) = b0_bvals;
    
    b9k_data(:,2:101) = data(:,201:300);
    b9k_bvecs(:,2:101) = bvecs(:,201:300);
    b9k_bvals(2:101) = bvals(201:300);
    
    save('b9k_data.mat','b9k_data')
    dlmwrite('b9k_bvecs.bvec',b9k_bvecs)
    dlmwrite('b9k_bvals.bval',b9k_bvals)
    % B12000
    b12k_data(:,1) = b0_data;
    b12k_bvecs(:,1) = b0_bvecs;
    b12k_bvals(1) = b0_bvals;
    
    b12k_data(:,2:101) = data(:,301:400);
    b12k_bvecs(:,2:101) = bvecs(:,301:400);
    b12k_bvals(2:101) = bvals(301:400);
    
    save('b12k_data.mat','b12k_data')
    dlmwrite('b12k_bvecs.bvec',b12k_bvecs)
    dlmwrite('b12k_bvals.bval',b12k_bvals)
    % Two Shell Sets
    % B3000 and B6000
    b3k_b6k_data(:,1) = b0_data;
    b3k_b6k_bvecs(:,1) = b0_bvecs;
    b3k_b6k_bvals(1) = b0_bvals;
    
    b3k_b6k_data(:,2:101) = data(:,1:100);
    b3k_b6k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b6k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b6k_data(:,102:201) = data(:,101:200);
    b3k_b6k_bvecs(:,102:201) = bvecs(:,101:200);
    b3k_b6k_bvals(102:201) = bvals(:,101:200);
    
    save('b3k_b6k_data.mat','b3k_b6k_data')
    dlmwrite('b3k_b6k_bvecs.bvec',b3k_b6k_bvecs)
    dlmwrite('b3k_b6k_bvals.bval',b3k_b6k_bvals)
    
    % B3000 and B9000
    b3k_b9k_data(:,1) = b0_data;
    b3k_b9k_bvecs(:,1) = b0_bvecs;
    b3k_b9k_bvals(1) = b0_bvals;
    
    b3k_b9k_data(:,2:101) = data(:,1:100);
    b3k_b9k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b9k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b9k_data(:,102:201) = data(:,201:300);
    b3k_b9k_bvecs(:,102:201) = bvecs(:,201:300);
    b3k_b9k_bvals(102:201) = bvals(:,201:300);
    
    save('b3k_b9k_data.mat','b3k_b9k_data')
    dlmwrite('b3k_b9k_bvecs.bvec',b3k_b9k_bvecs)
    dlmwrite('b3k_b9k_bvals.bval',b3k_b9k_bvals)
    
    
    % B3000 and B12000
    b3k_b12k_data(:,1) = b0_data;
    b3k_b12k_bvecs(:,1) = b0_bvecs;
    b3k_b12k_bvals(1) = b0_bvals;
    
    b3k_b12k_data(:,2:101) = data(:,1:100);
    b3k_b12k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b12k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b12k_data(:,102:201) = data(:,301:400);
    b3k_b12k_bvecs(:,102:201) = bvecs(:,301:400);
    b3k_b12k_bvals(102:201) = bvals(:,301:400);
    
    save('b3k_b12k_data.mat','b3k_b12k_data')
    dlmwrite('b3k_b12k_bvecs.bvec',b3k_b12k_bvecs)
    dlmwrite('b3k_b12k_bvals.bval',b3k_b12k_bvals)
    
    % B6000 and B9000
    b6k_b9k_data(:,1) = b0_data;
    b6k_b9k_bvecs(:,1) = b0_bvecs;
    b6k_b9k_bvals(1) = b0_bvals;
    
    b6k_b9k_data(:,2:101) = data(:,101:200);
    b6k_b9k_bvecs(:,2:101) = bvecs(:,101:200);
    b6k_b9k_bvals(2:101) = bvals(:,101:200);
    
    b6k_b9k_data(:,102:201) = data(:,201:300);
    b6k_b9k_bvecs(:,102:201) = bvecs(:,201:300);
    b6k_b9k_bvals(102:201) = bvals(:,201:300);
    
    save('b6k_b9k_data.mat','b6k_b9k_data')
    dlmwrite('b6k_b9k_bvecs.bvec',b6k_b9k_bvecs)
    dlmwrite('b6k_b9k_bvals.bval',b6k_b9k_bvals)
    
    
    
    % B6000 and B12000
    b6k_b12k_data(:,1) = b0_data;
    b6k_b12k_bvecs(:,1) = b0_bvecs;
    b6k_b12k_bvals(1) = b0_bvals;
    
    b6k_b12k_data(:,2:101) = data(:,101:200);
    b6k_b12k_bvecs(:,2:101) = bvecs(:,101:200);
    b6k_b12k_bvals(2:101) = bvals(:,101:200);
    
    b6k_b12k_data(:,102:201) = data(:,301:400);
    b6k_b12k_bvecs(:,102:201) = bvecs(:,301:400);
    b6k_b12k_bvals(102:201) = bvals(:,301:400);
    
    save('b6k_b12k_data.mat','b6k_b12k_data')
    dlmwrite('b6k_b12k_bvecs.bvec',b6k_b12k_bvecs)
    dlmwrite('b6k_b12k_bvals.bval',b6k_b12k_bvals)
    
    
    % B9000 and B12000
    b9k_b12k_data(:,1) = b0_data;
    b9k_b12k_bvecs(:,1) = b0_bvecs;
    b9k_b12k_bvals(1) = b0_bvals;
    
    b9k_b12k_data(:,2:101) = data(:,201:300);
    b9k_b12k_bvecs(:,2:101) = bvecs(:,201:300);
    b9k_b12k_bvals(2:101) = bvals(:,201:300);
    
    b9k_b12k_data(:,102:201) = data(:,301:400);
    b9k_b12k_bvecs(:,102:201) = bvecs(:,301:400);
    b9k_b12k_bvals(102:201) = bvals(:,301:400);
    
    save('b9k_b12k_data.mat','b9k_b12k_data')
    dlmwrite('b9k_b12k_bvecs.bvec',b9k_b12k_bvecs)
    dlmwrite('b9k_b12k_bvals.bval',b9k_b12k_bvals)
    
    % Triple Shell Sets
    % B3000, B6000 and B9000
    b3k_b6k_b9k_data(:,1) = b0_data;
    b3k_b6k_b9k_bvecs(:,1) = b0_bvecs;
    b3k_b6k_b9k_bvals(1) = b0_bvals;
    
    b3k_b6k_b9k_data(:,2:101) = data(:,1:100);
    b3k_b6k_b9k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b6k_b9k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b6k_b9k_data(:,102:201) = data(:,101:200);
    b3k_b6k_b9k_bvecs(:,102:201) = bvecs(:,101:200);
    b3k_b6k_b9k_bvals(102:201) = bvals(:,101:200);
    
    b3k_b6k_b9k_data(:,202:301) = data(:,201:300);
    b3k_b6k_b9k_bvecs(:,202:301) = bvecs(:,201:300);
    b3k_b6k_b9k_bvals(202:301) = bvals(:,201:300);
    
    save('b3k_b6k_b9k_data.mat','b3k_b6k_b9k_data')
    dlmwrite('b3k_b6k_b9k_bvecs.bvec',b3k_b6k_b9k_bvecs)
    dlmwrite('b3k_b6k_b9k_bvals.bval',b3k_b6k_b9k_bvals)
    
    
    % B3000, B9000 & B12000
    b3k_b9k_b12k_data(:,1) = b0_data;
    b3k_b9k_b12k_bvecs(:,1) = b0_bvecs;
    b3k_b9k_b12k_bvals(1) = b0_bvals;
    
    b3k_b9k_b12k_data(:,2:101) = data(:,1:100);
    b3k_b9k_b12k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b9k_b12k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b9k_b12k_data(:,102:201) = data(:,201:300);
    b3k_b9k_b12k_bvecs(:,102:201) = bvecs(:,201:300);
    b3k_b9k_b12k_bvals(102:201) = bvals(:,201:300);
    
    b3k_b9k_b12k_data(:,202:301) = data(:,301:400);
    b3k_b9k_b12k_bvecs(:,202:301) = bvecs(:,301:400);
    b3k_b9k_b12k_bvals(202:301) = bvals(:,301:400);
    
    save('b3k_b9k_b12k_data.mat','b3k_b9k_b12k_data')
    dlmwrite('b3k_b9k_b12k_bvecs.bvec',b3k_b9k_b12k_bvecs)
    dlmwrite('b3k_b9k_b12k_bvals.bval',b3k_b9k_b12k_bvals)
    
    % B6000, B9000 and B12000
    b6k_b9k_b12k_data(:,1) = b0_data;
    b6k_b9k_b12k_bvecs(:,1) = b0_bvecs;
    b6k_b9k_b12k_bvals(1) = b0_bvals;
    
    b6k_b9k_b12k_data(:,2:101) = data(:,101:200);
    b6k_b9k_b12k_bvecs(:,2:101) = bvecs(:,101:200);
    b6k_b9k_b12k_bvals(2:101) = bvals(:,101:200);
    
    b6k_b9k_b12k_data(:,102:201) = data(:,201:300);
    b6k_b9k_b12k_bvecs(:,102:201) = bvecs(:,201:300);
    b6k_b9k_b12k_bvals(102:201) = bvals(:,201:300);
    
    b6k_b9k_b12k_data(:,202:301) = data(:,301:400);
    b6k_b9k_b12k_bvecs(:,202:301) = bvecs(:,301:400);
    b6k_b9k_b12k_bvals(202:301) = bvals(:,301:400);
    
    save('b6k_b9k_b12k_data.mat','b6k_b9k_b12k_data')
    dlmwrite('b6k_b9k_b12k_bvecs.bvec',b6k_b9k_b12k_bvecs)
    dlmwrite('b6k_b9k_b12k_bvals.bval',b6k_b9k_b12k_bvals)
    
    % All Shells
    b3k_b6k_b9k_b12k_data(:,1) = b0_data;
    b3k_b6k_b9k_b12k_bvecs(:,1) = b0_bvecs;
    b3k_b6k_b9k_b12k_bvals(1) = b0_bvals;
    
    b3k_b6k_b9k_b12k_data(:,2:101) = data(:,1:100);
    b3k_b6k_b9k_b12k_bvecs(:,2:101) = bvecs(:,1:100);
    b3k_b6k_b9k_b12k_bvals(2:101) = bvals(:,1:100);
    
    b3k_b6k_b9k_b12k_data(:,102:201) = data(:,101:200);
    b3k_b6k_b9k_b12k_bvecs(:,102:201) = bvecs(:,101:200);
    b3k_b6k_b9k_b12k_bvals(102:201) = bvals(:,101:200);
    
    b3k_b6k_b9k_b12k_data(:,202:301) = data(:,201:300);
    b3k_b6k_b9k_b12k_bvecs(:,202:301) = bvecs(:,201:300);
    b3k_b6k_b9k_b12k_bvals(202:301) = bvals(:,201:300);
    
    b3k_b6k_b9k_b12k_data(:,302:401) = data(:,301:400);
    b3k_b6k_b9k_b12k_bvecs(:,302:401) = bvecs(:,301:400);
    b3k_b6k_b9k_b12k_bvals(302:401) = bvals(:,301:400);
    
    save('b3k_b6k_b9k_b12k_data.mat','b3k_b6k_b9k_b12k_data')
    dlmwrite('b3k_b6k_b9k_b12k_bvecs.bvec',b3k_b6k_b9k_b12k_bvecs)
    dlmwrite('b3k_b6k_b9k_b12k_bvals.bval',b3k_b6k_b9k_b12k_bvals)

end