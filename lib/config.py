import argparse

def params_setup(cmdline=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--init', type=bool, default=False,help='initialization the system')
  parser.add_argument('--model_name', type=str, default='17flowers',help='training sample name')
  parser.add_argument('--label_size', type=int, default=101,help='label size of training sample')
  parser.add_argument('--gpu_usage', type=float, default=0.9, help='tensorflow gpu memory fraction used')
  parser.add_argument('--img_size', type=int, default=164, help='image size, also model input size')
  parser.add_argument('--device_ip', type=list, default=['192.168.1.125',], help='all device IP list')
  parser.add_argument('--remote_ip', type=list, default=['127.0.0.1',8001], help='The IP of remote computer')
  parser.add_argument('--dial_num_list', type=list, default=((4,3),), help='the dial num of every device')
  parser.add_argument('--batch_size', type=int, default=100, help='batch size')


  if cmdline:
    args = parser.parse_args(cmdline)
  else:
    args = parser.parse_args()


  args.down_sampling = {str(n): 10000 for n in range(13)}

  return args

