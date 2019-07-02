from data_provider import mnist
# from data_provider import kth

datasets_map = {
    # 'kth': kth,
    'mnist': mnist
}

def data_provider(dataset_name, img_width, batch_size, 
    train_data_paths = None, valid_data_paths = None, test_data_paths = None,
    is_training = True, return_test = False):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_width: Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''

    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)

    if is_training:
        assert train_data_paths and valid_data_paths, \
            "The training and validation sets are not given in the training mode."
    
    if not is_training or return_test:
        assert test_data_paths, \
            "The test set is not given in the test mode."

    if train_data_paths:
        train_data_list = train_data_paths.split(',')
        train_input_param = {'paths': train_data_list,
                             'minibatch_size': batch_size,
                             'input_data_type': 'float32',
                             'is_output_sequence': True,
                             'name': dataset_name+' train iterator'}
        train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
        train_input_handle.begin(do_shuffle = True)

    if valid_data_paths:
        valid_data_list = valid_data_paths.split(',')
        valid_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'valid iterator'}
        valid_input_handle = datasets_map[dataset_name].InputHandle(valid_input_param)
        valid_input_handle.begin(do_shuffle = False)

    if test_data_paths:
        test_data_list = test_data_paths.split(',')
        test_input_param = {'paths': test_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle = False)

    if is_training:
        if return_test:
            return train_input_handle, valid_input_handle, test_input_handle
        else:
            return train_input_handle, valid_input_handle
    else:
        return test_input_handle
