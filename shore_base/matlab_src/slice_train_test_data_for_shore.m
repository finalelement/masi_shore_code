function slice_train_test_data_for_shore

    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\out_matrix.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\shore_coeffs_r8.mat')
    
    train_input_shore_r8 = input_shore(1:49995,:);
    train_output_shore = out_matrix(1:49995,:);
    
    test_input_shore_r8 = input_shore(49996:57267,:);
    test_output_shore = out_matrix(49996:57267,:);
    
    save('train_input_shore_r8.mat','train_input_shore_r8')
    save('train_output_shore.mat','train_output_shore')
    save('test_input_shore_r8.mat','test_input_shore_r8')
    save('test_output_shore.mat','test_output_shore')

end