#from simple_convnet3 import CNN,train_CNN
#”ネットワーク構造”に対応
from simple_convnet4 import CNN,train_CNN
#”学習画像読み込み”に対応
from load_dataset2 import Load_Dataset





if __name__ == "__main__":
    
#   画像の種類（RGB，medianなど）
    filter_num = "median41"
#   入力次元（RGB→3，メディアン→1，メディアン＋領域分割→13） 
    channel = 13

#   添え字：ネットワークを複数作成する場合に使用    
    str1 = str(1)
    filter_name = filter_num + "_" + str1
#   第1引数は画像データのディレクトリ名
    train,test = Load_Dataset("../95-5/"+str1+"/test/" ,"median41",channels = channel ,num = str1)

#   ネットワークの定義
    network = CNN()
#   GPU使用の可否負なら使用しない，0以上なら使用する（別途環境設定が必要）
    gpu_number = 0
#   学習の実行
    network = train_CNN(network, batchsize=50, gpu_id = gpu_number, max_epoch=50, train_dataset = train, test_dataset = test,number = filter_name)

    
    
    
