function simulate_other_b_value_signals_for_fod_signal

    fod_q_space_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_qspace.mat';
    fod_bvals_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvals.bval';
    fod_bvecs_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvecs.bvec';

    load(fod_q_space_path)
    fod_bval = dlmread(fod_bvals_path);
    fod_bvec = dlmread(fod_bvecs_path);
    
    new_fod_bval = zeros(1,401);
    new_fod_bvec = zeros(3,401);
    new_fod_q_space = zeros(length(new_q_fod),401);
    
    fod_q = new_q_fod(:,2:end);
    
    % Calculate FOD q-space ADC
    adc=log(fod_q)/(-3000);
    
    ln_s2 = adc * (3000 - 6000) + log(fod_q);
    ln_s3 = adc * (3000 - 9000) + log(fod_q);
    ln_s4 = adc * (3000 - 12000) + log(fod_q);
    
    s2 = exp(ln_s2);
    s3 = exp(ln_s3);
    s4 = exp(ln_s4);
    
    % Fill up the new bvals and the bvecs and the data
    new_fod_bval(1,1) = 0;
    new_fod_bval(1,2:101) = 3000;
    new_fod_bval(1,102:201) = 6000;
    new_fod_bval(1,202:301) = 9000;
    new_fod_bval(1,302:401) = 12000;
    
    new_fod_bvec(:,1) = 0;
    new_fod_bvec(:,2:101) = fod_bvec(:,2:101);
    new_fod_bvec(:,102:201) = fod_bvec(:,2:101);
    new_fod_bvec(:,202:301) = fod_bvec(:,2:101);
    new_fod_bvec(:,302:401) = fod_bvec(:,2:101);
    
    new_fod_q_space(:,1) = 1;
    new_fod_q_space(:,2:101) = new_q_fod(:,2:101);
    new_fod_q_space(:,102:201) = s2;
    new_fod_q_space(:,202:301) = s3;
    new_fod_q_space(:,302:401) = s4;
    
    save('decayed_fod.mat','new_fod_q_space')
    dlmwrite('decayed_fod.bval',new_fod_bval)
    dlmwrite('decayed_fod.bvec',new_fod_bvec)
    
    


end