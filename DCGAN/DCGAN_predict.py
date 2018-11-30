import argparse
import os

import chainer
from DCGAN import Generator

import numpy as np
from PIL import Image

def DCGAN_run(num):
    parser = argparse.ArgumentParser(description='Chainer: MNIST predicting CNN')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--rows', '-r', type=int, default=1,
                        help='Number of rows in the image')
    parser.add_argument('--cols', '-c', type=int, default=1,
                        help='Number of cols in the image')
    parser.add_argument('--out', '-o', default='generate_image',
                        help='Directory to output the result')
    parser.add_argument('--seed', type=int, default=num,
                        help='Random seed of z at visualization stage')
    args = parser.parse_args()

    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('# Number of rows in the image: {}'.format(args.rows))
    print('# Number of cols in the image: {}'.format(args.cols))
    print('')

    gen = Generator(n_hidden=args.n_hidden)
    chainer.serializers.load_npz('gen_epoch_3.npz', gen)

    np.random.seed(args.seed)
    n_images = args.rows * args.cols
    xp = gen.xp
    z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))

    x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed(args.seed)

    # gen_output_activation_func is sigmoid (0 ~ 1)
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    # gen output_activation_func is tanh (-1 ~ 1)
    # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((args.rows, args.cols, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((args.rows * H, args.cols * W))


    preview_path = "./generate_image/" +str(n)+'.png'.format(args.epoch)
    print(preview_path)

    Image.fromarray(x).save(preview_path)

if __name__ == '__main__':
   num = 0
   for n in range(10): 
       nun = num + 1
       DCGAN_run(n)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
