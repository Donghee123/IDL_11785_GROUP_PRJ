import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from Resnet import *
from test_sparsity import *
from utils import *
from model_train import *

if __name__ == "__main__":


  transform_train = transforms.Compose([  
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

  transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

  net = ResNet18()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  net = net.to(device)

  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

  checkpoint = torch.load('./models/Aug/ckpt_95.59.pth')

  net.load_state_dict(checkpoint['net'])
  criterion = nn.CrossEntropyLoss()
  #test(net, 1, "Aug", testloader, criterion, False)
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

  k = (sparsity.test(images = imageList,labels = labelList, epsilon=190, starting_k=120,iter_number=1024, CAM = True))/(32.0*32.0)

  print(k)