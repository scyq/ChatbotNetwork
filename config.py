class Config:
    epoches = 300
    cuda_device = "0"  # 哪几个GPU训练
    learning_rate = 1e-5
    save_per_epoch = 1  # 多少个epoch保存一个checkpoint
    batch_size = 12
    train_or_chat = True  # True代表训练, False代表聊天
    chat_model_ckpt = "./checkpoints/checkpoints_0505_0941_epoch4"
    continue_train = False  # 是否从断点开始训练
