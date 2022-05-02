import time

for epoch in range(0, 5):
    checkpoints_path = 'checkpoints/checkpoints_{time}_epoch{epoch}'.format(
        time=time.strftime('%m%d_%H%M'), epoch=epoch + 1)
    print(checkpoints_path)