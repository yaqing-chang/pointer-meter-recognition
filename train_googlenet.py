from __future__ import division, print_function, absolute_import

from lib import data_util
from lib.config import params_setup
from lib.googlenet import GoogLeNet
from datetime import datetime

import pickle, gzip
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17


#-------------------------------
#   Training
#-------------------------------
# scope_name, label_size = '17flowers', 17
# scope_name, label_size = '17portraits', 9
args = params_setup()
gnet = GoogLeNet(args=args) #img_size=227,  label_size=label_size, gpu_memory_fraction=0.4, scope_name=scope_name)
pkl_files = gnet.get_data(dirname=args.model_name, down_sampling=args.down_sampling)

epoch = 0
while True:
    for f in pkl_files:
        X, Y = pickle.load(gzip.open(f, 'rb'))
        gnet.fit(X, Y, n_epoch=1)
        print('[pkl_files] done with %s @ %s' % (f, datetime.now()))
    epoch += 1
    print("[Finish] all pkl_files been trained %i times." % epoch)
    
