class Config:
    epoches = 300
    cuda_device = "0, 1, 2"  # 使用哪几个GPU
    # cuda_device = "0"
    learning_rate = 1e-4
    save_per_epoch = 1  # 多少个epoch保存一个checkpoint
    train_dialog = False  # GRU中间的DNN是否训练
    batch_size = 72
    train_or_chat = True  # True代表训练, False代表聊天
    chat_model_ckpt = "./checkpoints/checkpoints_0507_1416_epoch20"
    continue_train = True  # 是否从断点开始训练
