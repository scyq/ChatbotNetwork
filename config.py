class Config:
    epoches = 300
    save_per_epoch = 1  # 多少个epoch保存一个checkpoint
    batch_size = 6
    train_or_chat = True  # True代表训练, False代表聊天
    chat_model_ckpt = "checkpoints/checkpoints_0502_1117_epoch1"
    continue_train = False  # 是否从断点开始训练
