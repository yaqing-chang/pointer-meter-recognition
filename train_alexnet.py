from lib.Alexnet import model2 as model


if __name__ == '__main__':
    models = model()
    dial_num = 2
    for i in range(dial_num):
        models.train_model(dial_type=i,epoch_num=4,batch_size=10)
    
    
    
