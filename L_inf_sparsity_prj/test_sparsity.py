import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
from cam import *
import os

class Test_Sparsity() :
    def __init__(self, transform = None, model=None,epsilon=None, starting_k=30,iter_number=30) :
        self.model = model
        self.epsilon = epsilon
        self.k = starting_k
        self.image_resolution = None
        self.transform = transform
        self.nTestCount = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_model(self,model) :
        self.model = model
    
    def set_epsilon(self,epsilon) :
        self.epsilon = epsilon

    def set_starting_k(self,k) :
        self.starting_k = k

    def set_k(set,k):
        self.k = k

    def set_image_resolution(self, resolution) :
        self.image_resolution = np.array(resolution)


    def make_l_inifity_attack_noise(self,image_shape,coordinates_list) :
        noise = np.zeros(image_shape)

        for coordinate in coordinates_list :
            low = -1 * self.epsilon
            high = 1 * self.epsilon
            noise[coordinate[0],coordinate[1]] = np.random.randint(low=low, high=high, size=3)
        
        return noise

    def Cv2Pillow(self, cvimage):
        color_coverted = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cvimage)
        return pil_image

    def Pillow2Cv(self, piimage):
        opencvImage = cv2.cvtColor(np.array(piimage), cv2.COLOR_RGB2BGR)
        return opencvImage

    def sampling_pixels_naive(self,image_shape,pixel_numbers) :
        height_pixels = np.random.randint(low=0, high=image_shape[0], size=pixel_numbers)
        width_pixels = np.random.randint(low=0, high=image_shape[1], size=pixel_numbers)

        #print(height_pixels)
        #print(width_pixels)

        sampled_pixels_coordinates = zip(height_pixels,width_pixels)

        #output [(cooridnates),(cooridnates),(cooridnates),(cooridnates)...]
        return list(sampled_pixels_coordinates)

    def get_cam_coordinates(self,image) :
        #TODO:
        transoformimage = self.transform(image)
        coordinates = getActivationPosition(self.model, image,transoformimage, self.nTestCount)
        
        return coordinates
    
    def sampling_pixels_CAM(self,image,pixel_numbers) :
        #height_pixels = np.random.randint(low=0, high=image_shape[0], size=pixel_numbers)
        #width_pixels = np.random.randint(low=0, high=image_shape[1], size=pixel_numbers)

        #print(height_pixels)
        #print(width_pixels)
        #sampled_pixels_coordinates = zip(height_pixels,width_pixels)


        CAM_coorinates_list = self.get_cam_coordinates(image)
        #TODO: RANDOM sampling from coordinates

        if pixel_numbers > len(CAM_coorinates_list) :
            pixel_numbers = len(CAM_coorinates_list)
        

        sampled_pixels_coordinates = random.sample(CAM_coorinates_list, pixel_numbers)
        
        
        #output [(cooridnates),(cooridnates),(cooridnates),(cooridnates)...]
        return list(sampled_pixels_coordinates)

    def sparsity_one_image(self,image : np.ndarray, label : int, CAM = False) :
        #print("input image is must in RGB. recommendation : use the method in this class -> load_image ")
        #print("return_True means model success // return_False means perbutation success")
        assert len(image.shape) == 3 ## not in proper shape
        assert image.shape[2] == 3 ## RGB image // not in grayscale

        if self.image_resolution == None :
            self.image_resolution = image.shape

        assert self.image_resolution == image.shape
        # whole images for sparsity test must have same input resolution 

        #opencv image 1장
        image_shape = image.shape

        
        if CAM == False :             
            coordinates_list = self.sampling_pixels_naive(image_shape=image_shape, pixel_numbers=self.k)
        else :
            coordinates_list = self.sampling_pixels_CAM(image=image, pixel_numbers=self.k)
        noise = self.make_l_inifity_attack_noise(image_shape=image_shape,coordinates_list=coordinates_list)
    
        fake_image = image + noise

        if not os.path.isdir('models'):
            os.mkdir('models')
        cv2.imwrite(f'./image/image_{self.nTestCount}.jpg', cv2.resize(image,(225,225)))
        cv2.imwrite(f'./noise/noise_{self.nTestCount}.jpg', cv2.resize(noise,(225,225)))
        cv2.imwrite(f'./fakeimage/fakeimage_{self.nTestCount}.jpg', cv2.resize(fake_image,(225,225)))

        self.nTestCount += 1

        np.clip(fake_image, 0, 255, out=fake_image)
        
        #BGR to RGB
        fake_image = fake_image.astype(np.uint8)        
        fake_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(fake_image)

        if self.transform is not None:
            inputimage = self.transform(img)
        
        fake_image = inputimage.type(torch.FloatTensor)
      
        fake_image = torch.unsqueeze(fake_image, dim=0)

        inference = self.model(fake_image.to(self.device))
        inference = inference.to('cpu')
        inference = inference.detach().numpy()
        
        y_hat = np.argmax(inference)
        if label == y_hat :
           return 1
        else :
           return 0


    def test(self, images : np.ndarray, labels : list or np.ndarray, iter_number : int, max_convergence_count = 5, starting_k = 30, epsilon = 10, CAM=False) :
        assert len(images) == len(labels) 

        self.starting_k = starting_k
        self.k = self.starting_k
        self.epsilon = epsilon

        convergence_count = 0
        before_iter_model_success = None
        model_sucess = None
        convergence_k = None

        for iter,i in enumerate(range(iter_number)) :
            success_num = 0
            for image,label in zip(images,labels) :
                success = self.sparsity_one_image(image,int(label),CAM)
                success_num = success_num + success

            print(success_num , len(images))
            # check model/attack success in this iter
            if ( success_num > int(len(images)/2) ) == True :
                self.k = self.k + 1
                print("attack fail!", "increase k! k =",self.k)
                model_sucess = True
            else :
                self.k = self.k - 1
                print("attack success!", "decrease k! k =",self.k)
                model_sucess = False

            # checking model/attack test is converged

            if iter == 0 :
                before_iter_model_success = model_sucess
                continue

            if model_sucess != before_iter_model_success :
                convergence_count = convergence_count + 1
            else :
                convergence_count = 0

            before_iter_model_success = model_sucess

            if convergence_count == 1 :
                if model_sucess == True : # attack fail
                    convergence_k = self.k - 1
                else : # attack success
                    convergence_k = self.k + 1
                

            print("convergence_count=",convergence_count)
            if convergence_count >= max_convergence_count :
                self.k = convergence_k
                print("convergence!, test finished in iter:",iter, 'k =',self.k)
                break

            if self.k == 0 :
                print("k=0", "test_finished in iter :",iter)
                return self.k
        
        print("test finished! maybe the model is not converged. please set iter_number more higher or set max_convergence_count lower", "k =", self.k)
        
        return self.k

            
    def load_image(self,image_path) :
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        print("RGB format image is return")
        print("height,width,color",image.shape)

        return image

    def load_images(self,image_paths:list) :
        image_array = list()
        for image_path in image_paths :
            image = self.load_image(image_path)
            image_array.append(image)

        
        return np.array(image_array)