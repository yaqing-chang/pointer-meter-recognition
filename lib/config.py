import argparse

def params_setup(cmdline=None):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--model_name', type=str, default='17flowers',help='training sample name')
  parser.add_argument('--label_size', type=int, default=101,help='label size of training sample')
  parser.add_argument('--gpu_usage', type=float, default=0.9, help='tensorflow gpu memory fraction used')
  parser.add_argument('--img_size', type=int, default=164, help='image size, also model input size')
  parser.add_argument('--device_ip', type=list, default=['192.168.1.125',], help='all device ip list')
  parser.add_argument('--dial_num_list', type=list, default=((4,0),), help='the dial num of every device')
  parser.add_argument('--batch_size', type=int, default=150, help='batch size')


  if cmdline:
    args = parser.parse_args(cmdline)
  else:
    args = parser.parse_args()


  args.down_sampling = {str(n): 10000 for n in range(13)}

  return args

