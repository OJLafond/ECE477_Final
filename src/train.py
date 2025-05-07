def run_training_scheme(X, y, scheme="A", checkpoint_dir="results", epochs_per_step=10):

    params_dict = {
        'A': {'init_size': 20, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.3, 'loops': 10, 'full_data': False},
        'B': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loops': 10, 'full_data': False},
        'C': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loops': 10, 'full_data': False},
    }

    assert scheme in params_dict, f"Unsupported scheme '{scheme}'"
    params = params_dict[scheme]

    save_dir = os.path.join(checkpoint_dir, f"scann_{scheme.lower()}")
    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.where(y_train == 'Stress', 1, 0)
    y_test = np.where(y_test == 'Stress', 1, 0)

    model = SDNN(in_num=X.shape[1], out_num=2,
                 init_size=params['init_size'],
                 max_size=params['max_size'],
                 batch_size=256,
                 scheme=scheme)

    model.structureInit(load=False, sparse=params['sparse'], ratio=params['sparse_ratio'])
    model.loadData(mode='train', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    model.train(epochs_per_step, save_dir)

    for i in range(params['loops']):
        print(f"\n[SCANN] === Loop {i+1}/{params['loops']} ===")
        model.addConnection(mode='grad',
                            percentile={'m2': 70, 'm1': 70, 'm3': 70, 'm4': 70},
                            full_data=params['full_data'])
        model.train(epochs_per_step, save_dir)

        if scheme == "A":
            model.cellDivision(mode='acti', full_data=params['full_data'])
            model.train(epochs_per_step, save_dir)
        elif scheme in ["B", "C"]:
            model.pruneConnections(prune_ratio=0.2)
            model.train(epochs_per_step, save_dir)

    print(f"[SCANN] Final accuracy for Scheme {scheme}:")
    acc = model.displayAcc()

    final_model_path = os.path.join(save_dir, f"SDNN_model_final.pth.tar")
    model.save_checkpoint({
        'epoch': model.epoch,
        'best_acc': model.best_acc,
        'now_acc': model.now_acc,
        'state_dict': {
            'w1': model.w1.data,
            'm1': model.m1.data,
            'w2': model.w2.data,
            'm2': model.m2.data,
            'w3': model.w3.data,
            'm3': model.m3.data,
            'w4': model.w4.data,
            'm4': model.m4.data,
            'b1': model.b1.data,
            'b2': model.b2.data,
        },
        'active_index': model.active_index,
        'optimizer': model.optimizer.state_dict(),
    }, is_best=False, folder_to_save=save_dir)

    return model, acc
