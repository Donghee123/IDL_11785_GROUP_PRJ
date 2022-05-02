import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from Resnet import *
from test_sparsity import *
from utils import *
from model_train import *
from utils import *

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Model Train code (None, Aug, FS, SPS)')

  parser.add_argument("-defence_name", type=str, help='Defence name only support (None, Aug, FS, SPS)', default='Aug')
  parser.add_argument("-use_cam", type=int, help='use cam 1, no cam 0', default=0)
    
  args = parser.parse_args()
  defence_name = args.defence_name
  use_cam = args.use_cam
  transform_train, transform_test= get_train_test_transfomes(defence_name)
  

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

  net = get_model(defence_name)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  net = net.to(device)

  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

  PATH = os.path.join('models', f'{defence_name}')
  files = os.listdir(PATH)
  
  checkpoint = torch.load(os.path.join(PATH, files[0]))

  net.load_state_dict(checkpoint['net'])
  criterion = nn.CrossEntropyLoss()

  testset = torchvision.datasets.CIFAR10(root='./data',download=True)

  extract_set = extract_image(100)

  transform_test = transforms.Compose([
      transforms.ToTensor(),    
  ])

  sparsity = Test_Sparsity(transform=transform_test, model=net)

  labelList = []
  imageList = []

  for index in range(10):
    for image in extract_set[index]:
      image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR) #opencv image로 변환 RGB to BGR
      imageList.append(image)
      labelList.append(index)

  if use_cam == 1:
    k = (sparsity.test(images = imageList,labels = labelList, epsilon=190, starting_k=120,iter_number=1024, CAM = True))/(32.0*32.0)
  else:
    k = (sparsity.test(images = imageList,labels = labelList, epsilon=190, starting_k=120,iter_number=1024, CAM = False))/(32.0*32.0)

  print(k)