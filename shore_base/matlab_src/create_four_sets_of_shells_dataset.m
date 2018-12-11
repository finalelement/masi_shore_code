function create_four_sets_of_shells_dataset

    % Load b3k,b9k,b3k&6k,b9k&b12k
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_shore_coeffs.mat')
    b3k = shore_coeffs;
    clear shore_coeffs
    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_shore_coeffs.mat')
    b9k = shore_coeffs;
    clear shore_coeffs
    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_shore_coeffs.mat')
    b3k_b6k = shore_coeffs;
    clear shore_coeffs
    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_b12k_shore_coeffs.mat')
    b9k_b12k = shore_coeffs;
    clear shore_coeffs
    
    % Load b6k & b3k_b6k_b9k for Testing
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_shore_coeffs.mat')
    b6k = shore_coeffs;
    clear shore_coeffs
    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_b9k_shore_coeffs.mat')
    b3k_b6k_b9k = shore_coeffs;
    clear shore_coeffs
    
    % Load Output Data
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\shore_coeffs_decayed_fod_r6.mat')
    fod_shore = output_shore;
    clear output_shore
    
    % Split into Train & Test Sets
    b3k_train = b3k(1:49995,:);
    b9k_train = b9k(1:49995,:);
    b3k_b6k_train = b3k_b6k(1:49995,:);
    b9k_b12k_train = b9k_b12k(1:49995,:);
    
    b3k_test = b3k(49996:end,:);
    b9k_test = b9k(49996:end,:);
    b6k_test = b6k(49996:end,:);
    b3k_b6k_test = b3k_b6k(49996:end,:);
    b9k_b12k_test = b9k_b12k(49996:end,:);
    b3k_b6k_b9k_test = b3k_b6k_b9k(49996:end,:);
    % Split output into Train and Test Sets
    fod_train = fod_shore(1:49995,:);
    fod_test = fod_shore(49996:end,:);
    
    % 4 Comb Train
    four_comb_ip_accum = [b3k_train;b9k_train;b3k_b6k_train;b9k_b12k_train];
    four_comb_op_accum = [fod_train;fod_train;fod_train;fod_train]; 
    
    sample_size = length(four_comb_ip_accum);
    
    rand_indices = randperm(sample_size,sample_size);
    
    four_comb_ip_train = four_comb_ip_accum(rand_indices,:);
    four_comb_op_train = four_comb_op_accum(rand_indices,:);
    
    % Save Paths
    % Base Saving Path
    save_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_organized_for_deep_learning\';
    b3k_test_path = sprintf('%sb3k_test.mat',save_path);
    b6k_test_path = sprintf('%sb6k_test.mat',save_path);
    b9k_test_path = sprintf('%sb9k_test.mat',save_path);
    b3k_b6k_test_path = sprintf('%sb3k_b6k_test.mat',save_path);
    b9k_b12k_test_path = sprintf('%sb9k_b12k_test.mat',save_path);
    b3k_b6k_b9k_test_path = sprintf('%sb3k_b6k_b9k_test.mat',save_path);
    
    b3k_train_ip_path = sprintf('%sb3k_train_ip.mat',save_path);
    b3k_train_op_path = sprintf('%sb3k_train_op.mat',save_path);
    
    four_comb_ip_path = sprintf('%sfour_comb_ip_train.mat',save_path);
    four_comb_op_path = sprintf('%sfour_comb_op_train.mat',save_path);
    
    save(b3k_test_path,'b3k_test')
    save(b6k_test_path,'b6k_test')
    save(b9k_test_path,'b9k_test')
    save(b3k_b6k_test_path,'b3k_b6k_test')
    save(b9k_b12k_test_path,'b9k_b12k_test')
    save(b3k_b6k_b9k_test_path,'b3k_b6k_b9k_test')
    
    save(b3k_train_ip_path,'b3k_train')
    save(b3k_train_op_path,'fod_train')
    
    save(four_comb_ip_path,'four_comb_ip_train')
    save(four_comb_op_path,'four_comb_op_train')
end